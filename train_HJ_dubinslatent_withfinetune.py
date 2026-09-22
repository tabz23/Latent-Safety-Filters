# train_HJ_dubinslatent_withfinetune.py
import argparse
import os
from pathlib import Path
import torch
import gymnasium as gym
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from omegaconf import OmegaConf
import yaml
import matplotlib.pyplot as plt
import wandb
from datetime import datetime
from copy import deepcopy
from tqdm import tqdm
import torch.nn.functional as F
from concurrent.futures import ThreadPoolExecutor
import functools

# Load DINO-WM via plan.load_model
from plan import load_model

# Underlying Dubins Gym env
from env.dubins.dubins import DubinsEnv
from gymnasium.spaces import Box

# Set up matplotlib config
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
os.makedirs(os.environ['MPLCONFIGDIR'], exist_ok=True)

def args_type(default):
    def parse_string(x):
        if default is None:
            return x
        if isinstance(default, bool):
            return bool(["False", "True"].index(x))
        if isinstance(default, int):
            return float(x) if ("e" in x or "." in x) else int(x)
        if isinstance(default, (list, tuple)):
            return tuple(args_type(default[0])(y) for y in x.split(","))
        return type(default)(x)

    def parse_object(x):
        if isinstance(default, (list, tuple)):
            return tuple(x)
        return x

    return lambda x: parse_string(x) if isinstance(x, str) else parse_object(x)

def get_args_and_merge_config():
    parser = argparse.ArgumentParser("DDPG HJ on DINO latent Dubins")
    parser.add_argument(
        "--dino_ckpt_dir", type=str,
        default="checkpoints/wm/dubins",
        help="Where to find the DINO-WM checkpoints"
    )
    parser.add_argument(
        "--config", type=str, default="train_HJ_configs.yaml",
        help="Path to your flat YAML of hyperparameters"
    )
    parser.add_argument(
        "--with_proprio", action="store_true",
        help="Flag to include proprioceptive information in latent encoding"
    )
    parser.add_argument(
        "--dino_encoder", type=str, default="dino",
        help="Which encoder to use: dino, r3m, vc1, resnet, dino_cls."
    )
    parser.add_argument(
        "--with_finetune", action="store_true",
        help="Flag to fine-tune the encoder backbone"
    )
    parser.add_argument(
        "--encoder_lr", type=float, default=1e-5,
        help="Learning rate for the encoder fine-tuning"
    )
    parser.add_argument('--critic-net', type=int, nargs=3, default=[512,512,512],
                    help='Hidden sizes for critic (expects 3 integers)')
    parser.add_argument('--control-net', type=int, nargs=3, default=[512,512,512],
                    help='Hidden sizes for control policy (expects 3 integers)')
    
    args, remaining = parser.parse_known_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    cfg_parser = argparse.ArgumentParser()
    for key, val in sorted(cfg.items()):
        arg_t = args_type(val)
        cfg_parser.add_argument(f"--{key}", type=arg_t, default=arg_t(val))
    cfg_args = cfg_parser.parse_args(remaining)

    for key, val in vars(cfg_args).items():
        setattr(args, key.replace("-", "_"), val)

    return args

def load_shared_world_model(ckpt_dir: str, device: str):
    ckpt_dir = Path(ckpt_dir)
    hydra_cfg = ckpt_dir / 'hydra.yaml'
    snapshot = ckpt_dir / 'checkpoints' / 'model_latest.pth'
    train_cfg = OmegaConf.load(str(hydra_cfg))
    num_action_repeat = train_cfg.num_action_repeat
    wm = load_model(snapshot, train_cfg, num_action_repeat, device=device)
    print(f"Loaded shared world model from {ckpt_dir}")
    return wm

class RawDubinsEnv(gym.Env):
    def __init__(self, device: str = None, with_proprio: bool = False):
        super().__init__()
        self.env = DubinsEnv()
        self.device = torch.device(device) if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.with_proprio = with_proprio
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def reset(self, state=None):
        if state is not None:
            reset_out = self.env.reset(state=state)
        else:
            reset_out = self.env.reset()
        obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
        return obs, {}

    def step(self, action):
        obs_out, _, done, info = self.env.step(action)
        terminated = done
        truncated = False
        obs = obs_out[0] if isinstance(obs_out, tuple) else obs_out
        h_s = info.get('h', 0.0) * 10 # Multiply by 10 to make HJ easier to learn
        return obs, h_s, terminated, truncated, info

class OptimizedReplayBuffer:
    """Optimized replay buffer using pre-allocated tensors"""
    def __init__(self, size: int, device: str, obs_example, action_dim):
        self.size = size
        self.device = torch.device(device)
        self.position = 0
        self.filled = 0
        
        # Extract dimensions from example observation
        if isinstance(obs_example, dict):
            visual_shape = obs_example['visual'].shape
            proprio_dim = obs_example['proprio'].shape[0]
        else:
            visual_shape = obs_example[0].shape
            proprio_dim = obs_example[1].shape[0]
        
        # Pre-allocate tensors on CPU to save GPU memory
        h, w, c = visual_shape
        # Store large visual buffers on CPU
        self.visual_buffer = torch.zeros((size, c, h, w), dtype=torch.float32, device='cpu')
        self.next_visual_buffer = torch.zeros((size, c, h, w), dtype=torch.float32, device='cpu')
        
        # Keep smaller tensors on GPU for faster access
        self.proprio_buffer = torch.zeros((size, proprio_dim), dtype=torch.float32, device=self.device)
        self.action_buffer = torch.zeros((size, action_dim), dtype=torch.float32, device=self.device)
        self.reward_buffer = torch.zeros((size, 1), dtype=torch.float32, device=self.device)
        self.next_proprio_buffer = torch.zeros((size, proprio_dim), dtype=torch.float32, device=self.device)
        self.done_buffer = torch.zeros((size, 1), dtype=torch.float32, device=self.device)

    def add(self, obs, act, rew, obs_next, done):
        # Extract visual and proprio
        if isinstance(obs, dict):
            visual = obs['visual']
            proprio = obs['proprio']
        else:
            visual, proprio = obs
            
        if isinstance(obs_next, dict):
            visual_next = obs_next['visual']
            proprio_next = obs_next['proprio']
        else:
            visual_next, proprio_next = obs_next
        
        # Convert and store directly in pre-allocated tensors
        self.visual_buffer[self.position] = torch.from_numpy(
            np.transpose(visual, (2, 0, 1)).astype(np.float32) / 255.0
        )
        self.proprio_buffer[self.position] = torch.from_numpy(proprio.astype(np.float32))
        self.action_buffer[self.position] = torch.from_numpy(act.astype(np.float32))
        self.reward_buffer[self.position] = rew
        self.next_visual_buffer[self.position] = torch.from_numpy(
            np.transpose(visual_next, (2, 0, 1)).astype(np.float32) / 255.0
        )
        self.next_proprio_buffer[self.position] = torch.from_numpy(proprio_next.astype(np.float32))
        self.done_buffer[self.position] = float(done)
        
        self.position = (self.position + 1) % self.size
        self.filled = min(self.filled + 1, self.size)

    def sample(self, batch_size: int):
        indices = torch.randint(0, self.filled, (batch_size,))
        
        # Move visual data to GPU during sampling
        visual_batch = self.visual_buffer[indices].to(self.device)
        next_visual_batch = self.next_visual_buffer[indices].to(self.device)
        
        return (
            (visual_batch, self.proprio_buffer[indices]),
            self.action_buffer[indices],
            self.reward_buffer[indices],
            (next_visual_batch, self.next_proprio_buffer[indices]),
            self.done_buffer[indices]
        )

    def __len__(self):
        return self.filled

def encode_batch_optimized(obs_batch, wm, device, with_proprio, requires_grad=False):
    """Optimized batch encoding - expects pre-processed tensors"""
    if isinstance(obs_batch, tuple):
        visual_batch, proprio_batch = obs_batch
        visual_batch = 2.0 * visual_batch - 1.0
    else:
        # Legacy path for list of observations
        visual_list = []
        proprio_list = []
        for obs in obs_batch:
            if isinstance(obs, dict):
                visual = obs['visual']
                proprio = obs['proprio']
            else:
                visual, proprio = obs
            
            # Check if visual data is already normalized [0,1] or raw [0,255]
            if visual.max() > 1.0:
                # Raw data [0,255] -> normalize to [0,1] first
                visual_np = np.transpose(visual, (2, 0, 1)).astype(np.float32) / 255.0
            else:
                # Already normalized [0,1] -> just transpose
                visual_np = np.transpose(visual, (2, 0, 1)).astype(np.float32)
            
            # APPLY WORLD MODEL NORMALIZATION: [0,1] -> [-1,1]
            visual_np = 2.0 * visual_np - 1.0
            
            visual_list.append(torch.from_numpy(visual_np))
            proprio_list.append(torch.from_numpy(proprio))
        visual_batch = torch.stack(visual_list).to(device)
        proprio_batch = torch.stack(proprio_list).to(device)
    
    # Add time dimension
    visual_batch = visual_batch.unsqueeze(1)
    proprio_batch = proprio_batch.unsqueeze(1)
    
    data = {'visual': visual_batch, 'proprio': proprio_batch}
    
    if requires_grad:
        lat = wm.encode_obs(data)
    else:
        with torch.no_grad():
            lat = wm.encode_obs(data)
    
    if with_proprio:
        z_vis = lat['visual'].reshape(lat['visual'].shape[0], -1)
        z_prop = lat['proprio'].squeeze(1)
        z = torch.cat([z_vis, z_prop], dim=-1)
    else:
        z = lat['visual'].reshape(lat['visual'].shape[0], -1)
    
    return z

def compute_hj_grid_vectorized(policy, helper_env, wm, theta, args, device):
    """Vectorized computation of HJ values for a grid at fixed theta"""
    xs = np.linspace(args.x_min, args.x_max, args.nx)
    ys = np.linspace(args.y_min, args.y_max, args.ny)
    
    # Create grid of states
    xx, yy = np.meshgrid(xs, ys, indexing='ij')
    states = np.stack([xx.ravel(), yy.ravel(), np.full(args.nx * args.ny, theta)], axis=1)
    
    # Batch process observations
    obs_list = []
    for state in states:
        obs_dict, _ = helper_env.env.reset(state=state)
        obs_list.append(obs_dict)
    
    # Process in larger batches
    batch_size = min(256, len(obs_list))  # Adjust based on GPU memory
    all_values = []
    
    with torch.no_grad():
        for i in range(0, len(obs_list), batch_size):
            batch = obs_list[i:i+batch_size]
            z = encode_batch_optimized(batch, wm, device, args.with_proprio, requires_grad=False)
            a = policy.actor_old(z)
            q_vals = policy.critic(z, a).squeeze(-1)
            all_values.append(q_vals.cpu().numpy())
    
    values = np.concatenate(all_values).reshape(args.nx, args.ny)
    return values

def plot_hj_optimized(policy, helper_env, wm, thetas, args, device):
    """Optimized HJ plotting using vectorized computation"""
    xs = np.linspace(args.x_min, args.x_max, args.nx)
    ys = np.linspace(args.y_min, args.y_max, args.ny)
    
    if len(thetas) == 1:
        fig1, axes1 = plt.subplots(1, 1, figsize=(6, 6))
        fig2, axes2 = plt.subplots(1, 1, figsize=(6, 6))
        axes1 = [axes1]
        axes2 = [axes2]
    else:
        fig1, axes1 = plt.subplots(len(thetas), 1, figsize=(6, 6*len(thetas)))
        fig2, axes2 = plt.subplots(len(thetas), 1, figsize=(6, 6*len(thetas)))
    
    for i, theta in enumerate(thetas):
        vals = compute_hj_grid_vectorized(policy, helper_env, wm, theta, args, device)
        
        axes1[i].imshow(
            (vals.T > 0),
            extent=(args.x_min, args.x_max, args.y_min, args.y_max),
            origin="lower",
            cmap='RdYlBu'
        )
        axes1[i].set_title(f"θ={theta:.2f} (safe mask)")
        axes1[i].set_xlabel("x")
        axes1[i].set_ylabel("y")
        
        im = axes2[i].imshow(
            vals.T,
            extent=(args.x_min, args.x_max, args.y_min, args.y_max),
            origin="lower",
            cmap='viridis'
        )
        axes2[i].set_title(f"θ={theta:.2f} (HJ value)")
        axes2[i].set_xlabel("x")
        axes2[i].set_ylabel("y")
        fig2.colorbar(im, ax=axes2[i])
    
    fig1.tight_layout()
    fig2.tight_layout()
    return fig1, fig2

class Actor(torch.nn.Module):
    def __init__(self, state_dim, action_dim, hidden_sizes, activation, max_action):
        super().__init__()
        self.net = self.build_net(state_dim, action_dim, hidden_sizes, activation)
        self.register_buffer('max_action', max_action)
        print("\nactor self.max_action ",self.max_action)

    def build_net(self, input_dim, output_dim, hidden_sizes, activation):
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_sizes:
            layers.append(torch.nn.Linear(prev_dim, hidden_dim))
            layers.append(getattr(torch.nn, activation)())
            prev_dim = hidden_dim
        layers.append(torch.nn.Linear(prev_dim, output_dim))
        layers.append(torch.nn.Tanh())
        return torch.nn.Sequential(*layers)

    def forward(self, state):
        return self.max_action * self.net(state)

class Critic(torch.nn.Module):
    def __init__(self, state_dim, action_dim, hidden_sizes, activation):
        super().__init__()
        self.net = self.build_net(state_dim + action_dim, 1, hidden_sizes, activation)

    def build_net(self, input_dim, output_dim, hidden_sizes, activation):
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_sizes:
            layers.append(torch.nn.Linear(prev_dim, hidden_dim))
            layers.append(getattr(torch.nn, activation)())
            prev_dim = hidden_dim
        layers.append(torch.nn.Linear(prev_dim, output_dim))
        return torch.nn.Sequential(*layers)

    def forward(self, state, action):
        return self.net(torch.cat([state, action], dim=-1))

class AvoidDDPGPolicy:
    def __init__(self, actor, actor_optim, critic, critic_optim, tau, gamma, exploration_noise, device, with_proprio, actor_gradient_steps=1, action_space=None):
        self.actor = actor
        self.actor_optim = actor_optim
        self.actor_old = deepcopy(actor)
        self.critic = critic
        self.critic_optim = critic_optim
        self.critic_old = deepcopy(critic)
        self.tau = tau
        self.gamma = gamma
        self._noise = exploration_noise  # Changed back to match original naming
        self.device = device
        self.with_proprio = with_proprio
        
        # Exact same parameters as original
        self.actor_gradient_steps = actor_gradient_steps #changed from 5
        self.new_expl = True
        self.warmup = False
        self._n_step = 1
        self._rew_norm = False
        self.action_low = action_space.low
        self.action_high = action_space.high
        print("\npolicy self.action_low ", self.action_low)
        print("\npolicy self.action_high ",self.action_high)
        print("\npolicy actor_gradient_steps ", self.actor_gradient_steps)

    def train(self):
        self.actor.train()
        self.critic.train()

    def eval(self):
        self.actor.eval()
        self.critic.eval()

    @torch.no_grad()
    def sync_weight(self):
        """Soft-update the weight for the target network - exact as original"""
        for target_param, param in zip(self.actor_old.parameters(), self.actor.parameters()):
            target_param.mul_(1 - self.tau).add_(param.data, alpha=self.tau)
        for target_param, param in zip(self.critic_old.parameters(), self.critic.parameters()):
            target_param.mul_(1 - self.tau).add_(param.data, alpha=self.tau)

    def _target_q(self, obs_next_batch, wm):
        """Predict the value of a state - exact as original _target_q method"""
        z_next = encode_batch_optimized(obs_next_batch, wm, self.device, self.with_proprio, requires_grad=False)
        with torch.no_grad():
            a_next = self.actor_old(z_next)
            target_q = self.critic_old(z_next, a_next)
        return target_q

    def _nstep_return_approximated_avoid_Bellman_equation(self, rew, terminal_value, gamma):
        """
        Implements the exact avoid Bellman equation from the original code.
        Convention: negative is unsafe
        V = min(l(x), V(x'))
        """
        target_shape = terminal_value.shape
        bsz = target_shape[0]
        
        # Take the worst between safety now and safety in the future
        target_q = gamma * torch.minimum(
            rew.reshape(bsz, 1),  # safety now
            terminal_value  # safety in the future
        ) + (1 - gamma) * rew.reshape(bsz, 1)  # discount toward safety now
        
        return target_q.reshape(target_shape)

    def compute_nstep_return(self, obs_batch, act_batch, rew_batch, obs_next_batch, done_batch, wm):
        """Compute the target q values using the avoid Bellman equation"""
        batch_size = rew_batch.shape[0]
        
        # Get target Q values
        target_q = self._target_q(obs_next_batch, wm)
        
        # Apply the avoid Bellman equation
        returns = self._nstep_return_approximated_avoid_Bellman_equation(rew_batch, target_q, self.gamma)
        
        return returns

    def learn(self, batch, wm, with_finetune, encoder_optim):
        """Update critic network and actor network - exact as original learn method"""
        obs_batch, act_batch, rew_batch, obs_next_batch, done_batch = batch
        requires_grad = with_finetune
        
        # Encode current states
        z = encode_batch_optimized(obs_batch, wm, self.device, self.with_proprio, requires_grad=requires_grad)
        
        # Compute target returns using avoid Bellman equation
        target_returns = self.compute_nstep_return(obs_batch, act_batch, rew_batch, obs_next_batch, done_batch, wm)
        
        # Critic update - using original's _mse_optimizer logic
        current_q = self.critic(z, act_batch).flatten()
        target_q = target_returns.flatten()
        td = current_q - target_q
        critic_loss = td.pow(2).mean()  # MSE loss as in original

        self.critic_optim.zero_grad()
        if with_finetune:
            encoder_optim.zero_grad()
        
        critic_loss.backward()
        self.critic_optim.step()
        
        grad_norm = 0.0
        if with_finetune:
            grad_norm = torch.nn.utils.clip_grad_norm_(wm.parameters(), max_norm=float('inf'))
            encoder_optim.step()

        # Actor update - exact as original: "update actor 5 times for each critic update"
        z_detached = z.detach()  # Detach to avoid gradients flowing to critic
        
        # LOG ACTOR ACTIONS BEFORE BACKPROP
        with torch.no_grad():
            current_actions = self.actor(z_detached)
            raw_network_output = self.actor.net(z_detached)  # Before max_action scaling
            
        if not self.warmup:
            # Store individual losses for logging (not averaging)
            actor_losses = []
            for _ in range(self.actor_gradient_steps):
                a = self.actor(z_detached)
                actor_loss = -self.critic(z_detached, a).mean()
                self.actor_optim.zero_grad()
                actor_loss.backward()
                self.actor_optim.step()
                actor_losses.append(actor_loss.item())
            # Return the last actor loss (not average) to match original
            final_actor_loss = actor_losses[-1]
        else:
            final_actor_loss = 0.0

        # Soft update the parameters - exact as original
        self.sync_weight()
        
        return {
            "loss/actor": final_actor_loss,
            "loss/critic": critic_loss.item(),
            "grad_norm": grad_norm if with_finetune else 0.0,
            # ACTION LOGGING
            "actions/mean": current_actions.mean().item(),
            "actions/std": current_actions.std().item(),
            "actions/min": current_actions.min().item(),
            "actions/max": current_actions.max().item(),
            "actions/raw_mean": raw_network_output.mean().item(),
            "actions/raw_std": raw_network_output.std().item(),
            "actions/raw_min": raw_network_output.min().item(),
            "actions/raw_max": raw_network_output.max().item(),
        }

    def exploration_noise(self, act, batch_obs):
        """
        Exact replication of original exploration_noise method.
        Note: In original, this is called with (act, batch) where batch contains obs
        """
        # Convert to numpy if needed
        if isinstance(act, torch.Tensor):
            act = act.cpu().numpy()
        
        # Apply Gaussian noise first (if enabled)
        if self._noise is not None and self._noise > 0:
            noise = np.random.normal(0, self._noise, act.shape)
            act = act + noise
        
        # Value-based exploration - exact as original
        if self.new_expl:
            rand_act = np.random.uniform(self.action_low, self.action_high, act.shape)
            
            # Encode observations for critic evaluation
            z = encode_batch_optimized(batch_obs, self.shared_wm, self.device, self.with_proprio, requires_grad=False)
            rand_act_tensor = torch.from_numpy(rand_act).float().to(self.device)
            
            with torch.no_grad():
                values = self.critic(z, rand_act_tensor).cpu().detach().numpy()
            
      
            # act = np.where(values < 0.0, rand_act, act)
        
        
            #"Where unsafe, keep policy action; where safe, use random action"
            
            # Where random actions would be unsafe (values < 0.0): stick with the policy action (act)
            # Where random actions would be safe (values >= 0.0): use the random action (rand_act)
            act = np.where(values < 0.0, act, rand_act)
            
            # noise = np.random.normal(0, self._noise, act.shape)
            # act = act + noise
            
        # Warmup override - exact as original
        if self.warmup:
            act = np.random.uniform(self.action_low, self.action_high, act.shape)

        return act

class ParallelEnvCollector:
    """Collects a fixed number of transitions from multiple envs in parallel"""
    def __init__(self, envs, policy, shared_wm, device, with_proprio):
        self.envs = envs
        self.policy = policy
        self.shared_wm = shared_wm
        self.device = device
        self.with_proprio = with_proprio
        self.num_envs = len(envs)
        # Store shared_wm in policy for exploration_noise method
        self.policy.shared_wm = shared_wm

    def collect_trajectories(self, buffer, max_steps: int=1000):
        """
        Collect up to `max_steps` transitions across all envs.
        Uses exact same exploration as original avoid_DDPGPolicy_annealing
        """
        # 1) Reset all envs and keep their latest obs
        obs_list = [env.reset()[0] for env in self.envs]
        done_flags = [False] * self.num_envs
        steps = 0

        while steps < max_steps:
            # 2) which envs are still active?
            active_idxs = [i for i, d in enumerate(done_flags) if not d]
            if not active_idxs:
                # all envs done → reset them if you want continuous data
                for i in range(self.num_envs):
                    obs_list[i] = self.envs[i].reset()[0]
                    done_flags[i] = False
                continue

            # 3) batch‑encode only active observations
            active_obs = [obs_list[i] for i in active_idxs]
            z = encode_batch_optimized(active_obs,
                                       self.shared_wm,
                                       self.device,
                                       self.with_proprio,
                                       requires_grad=False)
            
            # 4) Get actions from actor (without noise first)
            with torch.no_grad():
                raw_actions = self.policy.actor(z).cpu().numpy()

            # 5) Apply exact same exploration as original
            # Note: pass active_obs as the batch argument
            actions = self.policy.exploration_noise(raw_actions, active_obs)
            
            # Clip actions to valid range
            actions = np.clip(actions, 
                 self.envs[0].action_space.low, 
                 self.envs[0].action_space.high)

            # 6) step each active env and store transitions
            for idx, env_idx in enumerate(active_idxs):
                env = self.envs[env_idx]
                action = actions[idx]
                obs_next, rew, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                buffer.add(obs_list[env_idx], action, rew, obs_next, done)
                steps += 1
                obs_list[env_idx] = obs_next
                done_flags[env_idx] = done

                if steps >= max_steps:
                    break

def main():
    args = get_args_and_merge_config()
    args.critic_lr = float(args.critic_lr)
    args.actor_lr = float(args.actor_lr)
    args.tau = float(args.tau)
    args.gamma_pyhj = float(args.gamma_pyhj)
    args.exploration_noise = float(args.exploration_noise)
    args.step_per_epoch = int(args.step_per_epoch)
    args.training_num = int(args.training_num)
    args.total_episodes = int(args.total_episodes)
    args.batch_size_pyhj = int(args.batch_size_pyhj)
    args.buffer_size = int(args.buffer_size)
    if "full_scratch" in args.dino_encoder:
        args.dino_ckpt_dir = os.path.join(args.dino_ckpt_dir, "vc1")
        args.with_finetune = True
    else:
        args.dino_ckpt_dir = os.path.join(args.dino_ckpt_dir, args.dino_encoder)
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Enable cuDNN autotuner for better performance
    if device == 'cuda':
        torch.backends.cudnn.benchmark = True

    shared_wm = load_shared_world_model(args.dino_ckpt_dir, device)
    if args.with_finetune:
        shared_wm.train_encoder = True
        for p in shared_wm.parameters():
            p.requires_grad = True
        shared_wm.train()
    else:
        shared_wm.train_encoder = False
        for p in shared_wm.parameters():
            p.requires_grad = False
        shared_wm.eval()

    train_envs = [RawDubinsEnv(device=device, with_proprio=args.with_proprio) for _ in range(args.training_num)]
    test_envs = [RawDubinsEnv(device=device, with_proprio=args.with_proprio) for _ in range(args.test_num)]

    # Get dimensions
    dummy_obs, _ = train_envs[0].reset()
    state_dim = encode_batch_optimized([dummy_obs], shared_wm, device, args.with_proprio, requires_grad=False).shape[1]
    action_dim = train_envs[0].action_space.shape[0]
    max_action = torch.tensor(train_envs[0].action_space.high, device=device, dtype=torch.float32)

    # Initialize networks
    actor = Actor(state_dim, action_dim, args.control_net, args.actor_activation, max_action).to(device)
    critic = Critic(state_dim, action_dim, args.critic_net, args.critic_activation).to(device)

    actor_optim = torch.optim.AdamW(actor.parameters(), lr=args.actor_lr)
    critic_optim = torch.optim.AdamW(critic.parameters(), lr=args.critic_lr)

    if args.with_finetune:
        encoder_optim = torch.optim.AdamW(shared_wm.parameters(), lr=args.encoder_lr)
    else:
        encoder_optim = None

    policy = AvoidDDPGPolicy(
        actor=actor,
        actor_optim=actor_optim,
        critic=critic,
        critic_optim=critic_optim,
        tau=args.tau,
        gamma=args.gamma_pyhj,
        exploration_noise=args.exploration_noise,
        device=device,
        with_proprio=args.with_proprio,
        actor_gradient_steps=getattr(args, 'actor_gradient_steps', 1),
        action_space=train_envs[0].action_space
    )
    print("args.with_proprio",args.with_proprio)
    # Use optimized replay buffer
    buffer = OptimizedReplayBuffer(args.buffer_size, device, dummy_obs, action_dim)

    # Initialize parallel collector
    collector = ParallelEnvCollector(train_envs, policy, shared_wm, device, args.with_proprio)

    timestamp = datetime.now().strftime("%m%d_%H%M")
    log_dir = Path(f"runs/ddpg_hj_latent/{args.dino_encoder}-{timestamp}/")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    wandb.init(
        project=f"ddpg-hj-latent-dubins",
        name=f"ddpg-{args.dino_encoder}-{timestamp}",
        config=vars(args)
    )
    writer = SummaryWriter(log_dir=log_dir / "logs")

    thetas = [0.0, np.pi/4, np.pi/2, 3*np.pi/4]
    helper_env = RawDubinsEnv(device=device, with_proprio=args.with_proprio)

    # Warm up buffer more efficiently
    print(f"Warming up replay buffer (need ≥ {args.batch_size_pyhj} samples)...")
    while len(buffer) < 5000:
        collector.collect_trajectories(buffer)
    print(f"Replay buffer: {len(buffer)} samples.")

    for epoch in range(1, args.total_episodes + 1):
        print(f"\n=== Epoch {epoch}/{args.total_episodes} ===")
        
        # Collect data in parallel
        collector.collect_trajectories(buffer)
        
        # Train with progress bar
        pbar = tqdm(range(args.step_per_epoch), desc=f"Epoch {epoch}", unit="step")
        for step in pbar:
            batch = buffer.sample(args.batch_size_pyhj)
            metrics = policy.learn(batch, shared_wm, args.with_finetune, encoder_optim)
            
            pbar.set_postfix({
                "actor_loss": f"{metrics['loss/actor']:.8f}",
                "critic_loss": f"{metrics['loss/critic']:.8f}",
                "enc_grad": f"{metrics['grad_norm']:.8f}",
                "buffer": len(buffer)
            })
            wandb.log(metrics, step=epoch * args.step_per_epoch + step)
        pbar.close()

        # Plot HJ using optimized function
        fig1, fig2 = plot_hj_optimized(policy, helper_env, shared_wm, thetas, args, device)
        wandb.log({
            "HJ_latent/binary": wandb.Image(fig1),
            "HJ_latent/continuous": wandb.Image(fig2),
        })
        plt.close(fig1)
        plt.close(fig2)

        # Save checkpoints
        epoch_dir = log_dir / f"epoch_{epoch}"
        epoch_dir.mkdir(exist_ok=True)
        torch.save(policy.actor.state_dict(), epoch_dir / "actor.pth")
        torch.save(policy.critic.state_dict(), epoch_dir / "critic.pth")
        if args.with_finetune and epoch==args.total_episodes:
            torch.save(shared_wm.state_dict(), epoch_dir / "wm.pth")

    print("Training complete.")

if __name__ == "__main__":
    main()
    
    
''' 
NOTE
currently loss to encoder backpropped only through the critic not also through the actor
'''


'''add actor gradient steps to the conf'''

# python train_HJ_dubinslatent_withfinetune.py  --config train_HJ_configs.yaml --dino_encoder vc1  --with_finetune --encoder_lr 1e-6 --nx 50 --ny 50 --step-per-epoch 200 --total-episodes 200 --batch_size-pyhj 64 --gamma-pyhj 0.99 --actor-gradient-steps 2
# python train_HJ_dubinslatent_withfinetune.py  --config train_HJ_configs.yaml --dino_encoder vc1  --nx 50 --ny 50 --step-per-epoch 200 --total-episodes 200 --batch_size-pyhj 64 --gamma-pyhj 0.99 --actor-gradient-steps 2

# python train_HJ_dubinslatent_withfinetune.py  --config train_HJ_configs.yaml --dino_encoder r3m  --with_finetune --encoder_lr 1e-6 --nx 50 --ny 50 --step-per-epoch 200 --total-episodes 200 --batch_size-pyhj 64 --gamma-pyhj 0.99 --actor-gradient-steps 2
# python train_HJ_dubinslatent_withfinetune.py  --config train_HJ_configs.yaml --dino_encoder r3m  --nx 50 --ny 50 --step-per-epoch 200 --total-episodes 200 --batch_size-pyhj 64 --gamma-pyhj 0.99 --actor-gradient-steps 2

# python train_HJ_dubinslatent_withfinetune.py  --config train_HJ_configs.yaml --dino_encoder resnet  --with_finetune --encoder_lr 1e-6 --nx 50 --ny 50 --step-per-epoch 200 --total-episodes 200 --batch_size-pyhj 64 --gamma-pyhj 0.99 --actor-gradient-steps 2
# python train_HJ_dubinslatent_withfinetune.py  --config train_HJ_configs.yaml --dino_encoder resnet  --nx 50 --ny 50 --step-per-epoch 200 --total-episodes 200 --batch_size-pyhj 64 --gamma-pyhj 0.99 --actor-gradient-steps 2

# python train_HJ_dubinslatent_withfinetune.py  --config train_HJ_configs.yaml --dino_encoder dino_cls  --with_finetune --encoder_lr 1e-6 --nx 50 --ny 50 --step-per-epoch 200 --total-episodes 200 --batch_size-pyhj 64 --gamma-pyhj 0.99 --actor-gradient-steps 2
# python train_HJ_dubinslatent_withfinetune.py  --config train_HJ_configs.yaml --dino_encoder dino_cls  --nx 50 --ny 50 --step-per-epoch 200 --total-episodes 200 --batch_size-pyhj 64 --gamma-pyhj 0.99 --actor-gradient-steps 2

# python train_HJ_dubinslatent_withfinetune.py  --config train_HJ_configs.yaml --dino_encoder scratch  --with_finetune --encoder_lr 1e-6 --nx 50 --ny 50 --step-per-epoch 200 --total-episodes 200 --batch_size-pyhj 64 --gamma-pyhj 0.99 --actor-gradient-steps 2
# python train_HJ_dubinslatent_withfinetune.py  --config train_HJ_configs.yaml --dino_encoder scratch  --nx 50 --ny 50 --step-per-epoch 200 --total-episodes 200 --batch_size-pyhj 64 --gamma-pyhj 0.99 --actor-gradient-steps 2

# python train_HJ_dubinslatent_withfinetune.py  --config train_HJ_configs.yaml --dino_encoder dino  --with_finetune --encoder_lr 1e-6 --nx 50 --ny 50 --step-per-epoch 200 --total-episodes 200 --batch_size-pyhj 64 --gamma-pyhj 0.99 --actor-gradient-steps 2
# python train_HJ_dubinslatent_withfinetune.py  --config train_HJ_configs.yaml --dino_encoder dino  --nx 50 --ny 50 --step-per-epoch 200 --total-episodes 200 --batch_size-pyhj 64 --gamma-pyhj 0.99 --actor-gradient-steps 2





# python train_HJ_dubinslatent_withfinetune.py --dino_ckpt_dir "checkpoints/wm/dubins"  --config train_HJ_configs.yaml --dino_encoder dino_cls  --nx 50 --ny 50 --step-per-epoch 200 --total-episodes 200 --batch_size-pyhj 64 --gamma-pyhj 0.99 --actor-gradient-steps 2 --critic_net 512 512 512 --control_net 512 512 512



# python train_HJ_dubinslatent_withfinetune.py --dino_ckpt_dir "checkpoints/wm/dubins"  --config train_HJ_configs.yaml --dino_encoder dino_cls --nx 50 --ny 50 --step-per-epoch 200 --total-episodes 100 --batch_size-pyhj 64 --gamma-pyhj 0.99 --actor-gradient-steps 2 --critic-net 128 128 128  --control-net 128 128 128  --with_finetune --encoder_lr 1e-6