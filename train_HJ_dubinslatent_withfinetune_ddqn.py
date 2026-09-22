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
    parser = argparse.ArgumentParser("DDQN HJ on DINO latent Dubins")
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
    # COMMENTED OUT: Actor network no longer needed for DDQN
    # parser.add_argument('--control-net', type=int, nargs=3, default=[512,512,512],
    #                 help='Hidden sizes for control policy (expects 3 integers)')
    
    # ADDED: Number of discrete actions for DDQN
    parser.add_argument('--num_actions', type=int, default=3,
                    help='Number of discrete actions spanning -1 to 1')
    
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
        # MODIFIED: Store discrete action indices instead of continuous actions
        self.action_buffer = torch.zeros((size, 1), dtype=torch.long, device=self.device)  # Changed to long for indices
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
        # MODIFIED: Store action index instead of continuous action
        self.action_buffer[self.position] = act  # act should now be an integer index
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
            # MODIFIED: Get Q-values for all actions and take the max (best action)
            q_vals = policy.critic(z)  # Shape: (batch_size, num_actions)
            max_q_vals = q_vals.max(dim=1)[0]  # Take max over actions
            all_values.append(max_q_vals.cpu().numpy())
    
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

# COMMENTED OUT: Actor class no longer needed for DDQN
# class Actor(torch.nn.Module):
#     def __init__(self, state_dim, action_dim, hidden_sizes, activation, max_action):
#         super().__init__()
#         self.net = self.build_net(state_dim, action_dim, hidden_sizes, activation)
#         self.register_buffer('max_action', max_action)
#         print("\nactor self.max_action ",self.max_action)

#     def build_net(self, input_dim, output_dim, hidden_sizes, activation):
#         layers = []
#         prev_dim = input_dim
#         for hidden_dim in hidden_sizes:
#             layers.append(torch.nn.Linear(prev_dim, hidden_dim))
#             layers.append(getattr(torch.nn, activation)())
#             prev_dim = hidden_dim
#         layers.append(torch.nn.Linear(prev_dim, output_dim))
#         layers.append(torch.nn.Tanh())
#         return torch.nn.Sequential(*layers)

#     def forward(self, state):
#         return self.max_action * self.net(state)

class Critic(torch.nn.Module):
    # MODIFIED: Critic now outputs Q-values for all discrete actions
    def __init__(self, state_dim, num_actions, hidden_sizes, activation):
        super().__init__()
        # Output num_actions Q-values instead of single value
        self.net = self.build_net(state_dim, num_actions, hidden_sizes, activation)

    def build_net(self, input_dim, output_dim, hidden_sizes, activation):
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_sizes:
            layers.append(torch.nn.Linear(prev_dim, hidden_dim))
            layers.append(getattr(torch.nn, activation)())
            prev_dim = hidden_dim
        layers.append(torch.nn.Linear(prev_dim, output_dim))
        return torch.nn.Sequential(*layers)

    def forward(self, state):
        # MODIFIED: Return Q-values for all actions, not Q(s,a) for specific action
        return self.net(state)

# ADDED: Helper function to convert between action indices and continuous actions
def action_index_to_continuous(action_indices, num_actions, action_low=-1.0, action_high=1.0):
    """Convert discrete action indices to continuous action values"""
    if isinstance(action_indices, torch.Tensor):
        action_indices = action_indices.cpu().numpy()
    
    # Map indices [0, num_actions-1] to continuous values [action_low, action_high]
    continuous_actions = action_low + (action_indices / (num_actions - 1)) * (action_high - action_low)
    return continuous_actions

def continuous_to_action_index(continuous_actions, num_actions, action_low=-1.0, action_high=1.0):
    """Convert continuous action values to discrete action indices"""
    # Clamp actions to valid range
    continuous_actions = np.clip(continuous_actions, action_low, action_high)
    
    # Map continuous values [action_low, action_high] to indices [0, num_actions-1]
    normalized = (continuous_actions - action_low) / (action_high - action_low)
    indices = (normalized * (num_actions - 1)).round().astype(int)
    return indices

class AvoidDDQNPolicy:
    # MODIFIED: Replaced DDPG with DDQN - removed actor, changed to discrete actions
    def __init__(self, critic, critic_optim, tau, gamma, exploration_noise, device, with_proprio, 
                 num_actions=20, action_space=None):
        # Removed actor components
        self.critic = critic
        self.critic_optim = critic_optim
        self.critic_old = deepcopy(critic)
        self.tau = tau
        self.gamma = gamma
        self._noise = exploration_noise
        self.device = device
        self.with_proprio = with_proprio
        
        # ADDED: Discrete action parameters
        self.num_actions = num_actions
        self.action_low = action_space.low[0]  # Assuming 1D action space
        self.action_high = action_space.high[0]
        
        # Generate discrete action grid
        self.action_grid = np.linspace(self.action_low, self.action_high, self.num_actions)
        
        # Exploration parameters
        self.new_expl = True
        self.warmup = False
        self._n_step = 1
        self._rew_norm = False
        
        print(f"\nDDQN with {self.num_actions} discrete actions from {self.action_low} to {self.action_high}")
        print(f"Action grid: {self.action_grid}")

    def train(self):
        self.critic.train()

    def eval(self):
        self.critic.eval()

    @torch.no_grad()
    def sync_weight(self):
        """Soft-update the weight for the target network"""
        for target_param, param in zip(self.critic_old.parameters(), self.critic.parameters()):
            target_param.mul_(1 - self.tau).add_(param.data, alpha=self.tau)

    def _target_q(self, obs_next_batch, wm):
        """Predict the maximum Q-value for next states"""
        z_next = encode_batch_optimized(obs_next_batch, wm, self.device, self.with_proprio, requires_grad=False)
        with torch.no_grad():
            target_q_all = self.critic_old(z_next)  # Shape: (batch_size, num_actions)
            target_q = target_q_all.max(dim=1, keepdim=True)[0]  # Take max over actions
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
        """Update critic network for DDQN"""
        obs_batch, act_batch, rew_batch, obs_next_batch, done_batch = batch
        requires_grad = with_finetune
        
        # Encode current states
        z = encode_batch_optimized(obs_batch, wm, self.device, self.with_proprio, requires_grad=requires_grad)
        
        # Compute target returns using avoid Bellman equation
        target_returns = self.compute_nstep_return(obs_batch, act_batch, rew_batch, obs_next_batch, done_batch, wm)
        
        # MODIFIED: Critic update for DDQN - select Q-values for taken actions
        q_all = self.critic(z)  # Shape: (batch_size, num_actions)
        current_q = q_all.gather(1, act_batch)  # Select Q-values for taken actions
        target_q = target_returns
        
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

        # REMOVED: Actor update - not needed for DDQN
        
        # Soft update the parameters
        self.sync_weight()
        
        # ADDED: Log Q-value statistics instead of actor actions
        with torch.no_grad():
            q_values = self.critic(z.detach())
            max_q_indices = q_values.argmax(dim=1)
            best_actions = action_index_to_continuous(max_q_indices, self.num_actions, self.action_low, self.action_high)
        
        return {
            "loss/critic": critic_loss.item(),
            "grad_norm": grad_norm if with_finetune else 0.0,
            # MODIFIED: Log best actions from Q-network instead of actor actions
            "actions/mean": np.mean(best_actions),
            "actions/std": np.std(best_actions),
            "actions/min": np.min(best_actions),
            "actions/max": np.max(best_actions),
            "q_values/mean": q_values.mean().item(),
            "q_values/std": q_values.std().item(),
            "q_values/min": q_values.min().item(),
            "q_values/max": q_values.max().item(),
        }

    def exploration_noise(self, z_batch):
        """
        MODIFIED: Epsilon-greedy exploration for DDQN instead of noise-based exploration
        """
        batch_size = z_batch.shape[0]
        
        # Get Q-values for all actions
        with torch.no_grad():
            q_values = self.critic(z_batch)  # Shape: (batch_size, num_actions)
        
        # Epsilon-greedy action selection
        epsilon = self._noise if not self.warmup else 1.0  # Use noise parameter as epsilon
        
        action_indices = []
        for i in range(batch_size):
            if np.random.random() < epsilon:
                # Random action
                action_idx = np.random.randint(0, self.num_actions)
            else:
                # Greedy action
                action_idx = q_values[i].argmax().item()
            action_indices.append(action_idx)
        
        # Value-based exploration - same logic as original but adapted for discrete actions
        if self.new_expl:
            for i in range(batch_size):
                rand_action_idx = np.random.randint(0, self.num_actions)
                rand_q_value = q_values[i, rand_action_idx].item()
                
                # Where random actions would be unsafe (values < 0.0): stick with the greedy action
                # Where random actions would be safe (values >= 0.0): use the random action
                if rand_q_value >= 0.0:
                    action_indices[i] = rand_action_idx
        
        return np.array(action_indices)

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
        MODIFIED: Collect trajectories using DDQN discrete action selection
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
            
            # 4) MODIFIED: Get discrete action indices using epsilon-greedy
            action_indices = self.policy.exploration_noise(z)
            
            # 5) MODIFIED: Convert action indices to continuous actions for environment
            continuous_actions = action_index_to_continuous(
                action_indices, 
                self.policy.num_actions, 
                self.policy.action_low, 
                self.policy.action_high
            )

            # 6) step each active env and store transitions
            for idx, env_idx in enumerate(active_idxs):
                env = self.envs[env_idx]
                action_continuous = continuous_actions[idx]
                action_idx = action_indices[idx]
                
                obs_next, rew, terminated, truncated, info = env.step([action_continuous])
                done = terminated or truncated

                # MODIFIED: Store action index instead of continuous action
                buffer.add(obs_list[env_idx], action_idx, rew, obs_next, done)
                steps += 1
                obs_list[env_idx] = obs_next
                done_flags[env_idx] = done

                if steps >= max_steps:
                    break
def plot_best_actions_grid(policy, helper_env, wm, thetas, args, device):
    """Plot the best action (argmax Q) at each state in a grid"""
    # Fixed resolution for action plotting
    nx_actions = 30
    ny_actions = 30
    
    xs = np.linspace(args.x_min, args.x_max, nx_actions)
    ys = np.linspace(args.y_min, args.y_max, ny_actions)
    
    if len(thetas) == 1:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        axes = [axes]
    else:
        fig, axes = plt.subplots(len(thetas), 2, figsize=(14, 6*len(thetas)))
    
    for i, theta in enumerate(thetas):
        # Create grid of states
        xx, yy = np.meshgrid(xs, ys, indexing='ij')
        states = np.stack([xx.ravel(), yy.ravel(), np.full(nx_actions * ny_actions, theta)], axis=1)
        
        # Batch process observations
        obs_list = []
        for state in states:
            obs_dict, _ = helper_env.env.reset(state=state)
            obs_list.append(obs_dict)
        
        # Process in batches
        batch_size = min(256, len(obs_list))
        all_best_actions = []
        all_q_values = []
        
        with torch.no_grad():
            for j in range(0, len(obs_list), batch_size):
                batch = obs_list[j:j+batch_size]
                z = encode_batch_optimized(batch, wm, device, args.with_proprio, requires_grad=False)
                
                # Get Q-values for all actions
                q_vals = policy.critic(z)  # Shape: (batch_size, num_actions)
                
                # Get best action indices
                best_action_indices = q_vals.argmax(dim=1)  # (batch_size,)
                
                # Convert to continuous actions
                best_continuous_actions = action_index_to_continuous(
                    best_action_indices.cpu().numpy(),
                    policy.num_actions,
                    policy.action_low,
                    policy.action_high
                )
                
                all_best_actions.extend(best_continuous_actions)
                all_q_values.append(q_vals.cpu().numpy())
        
        # Reshape for plotting
        best_actions_grid = np.array(all_best_actions).reshape(nx_actions, ny_actions)
        all_q_values_array = np.concatenate(all_q_values).reshape(nx_actions, ny_actions, policy.num_actions)
        
        # Plot 1: Best action at each state
        im1 = axes[i][0].imshow(
            best_actions_grid.T,
            extent=(args.x_min, args.x_max, args.y_min, args.y_max),
            origin="lower",
            cmap='RdBu',  # Red = -1 (left), Blue = 1 (right), White = 0 (straight)
            vmin=-1.0,
            vmax=1.0
        )
        axes[i][0].set_title(f"θ={theta:.2f} - Best Action (argmax Q)")
        axes[i][0].set_xlabel("x")
        axes[i][0].set_ylabel("y")
        fig.colorbar(im1, ax=axes[i][0], label="Action Value")
        
        # Add quiver plot to show action directions
        skip = 3  # Show arrows every 3rd grid point for clarity
        X, Y = xx[::skip, ::skip], yy[::skip, ::skip]
        U = np.cos(theta) * np.ones_like(X)  # Forward direction based on theta
        V = np.sin(theta) * np.ones_like(Y)
        # Rotate by action value (steering angle)
        actions_subset = best_actions_grid[::skip, ::skip]
        U_rot = U * np.cos(actions_subset) - V * np.sin(actions_subset)
        V_rot = U * np.sin(actions_subset) + V * np.cos(actions_subset)
        
        axes[i][0].quiver(X, Y, U_rot, V_rot, 
                         actions_subset,  # Color by action value
                         cmap='RdBu', alpha=0.7, scale=30)
        
        # Plot 2: Q-value difference (to see confidence in action selection)
        # Show difference between best and second-best Q-value
        q_sorted = np.sort(all_q_values_array, axis=2)
        q_diff = q_sorted[:, :, -1] - q_sorted[:, :, -2] if policy.num_actions > 1 else q_sorted[:, :, -1]
        
        im2 = axes[i][1].imshow(
            q_diff.T,
            extent=(args.x_min, args.x_max, args.y_min, args.y_max),
            origin="lower",
            cmap='viridis'
        )
        axes[i][1].set_title(f"θ={theta:.2f} - Q-value Confidence (Q_best - Q_second)")
        axes[i][1].set_xlabel("x")
        axes[i][1].set_ylabel("y")
        fig.colorbar(im2, ax=axes[i][1], label="Q-value Difference")
    
    fig.tight_layout()
    return fig


def plot_all_q_values_comparison(policy, helper_env, wm, theta, position, args, device):
    """
    Plot Q-values for all actions at a specific position
    Useful for debugging what the network thinks about each action
    """
    # Reset to the specific state
    state = np.array([position[0], position[1], theta])
    obs_dict, _ = helper_env.env.reset(state=state)
    
    # Encode observation
    z = encode_batch_optimized([obs_dict], wm, device, args.with_proprio, requires_grad=False)
    
    with torch.no_grad():
        q_vals = policy.critic(z).squeeze().cpu().numpy()  # (num_actions,)
    
    # Create bar plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    action_values = policy.action_grid  # The actual continuous action values
    colors = ['red' if q < 0 else 'green' for q in q_vals]
    
    bars = ax.bar(range(policy.num_actions), q_vals, color=colors, alpha=0.7)
    
    # Add action values as x-tick labels
    ax.set_xticks(range(policy.num_actions))
    ax.set_xticklabels([f"{a:.2f}" for a in action_values], rotation=45)
    
    ax.set_xlabel("Action Value (Steering)")
    ax.set_ylabel("Q-value")
    ax.set_title(f"Q-values at position ({position[0]:.1f}, {position[1]:.1f}), θ={theta:.2f}")
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.5, label='Safety Threshold')
    
    # Highlight the best action
    best_idx = np.argmax(q_vals)
    ax.scatter(best_idx, q_vals[best_idx], color='blue', s=100, zorder=5, 
              label=f'Best Action: {action_values[best_idx]:.2f}')
    
    # Add grid
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Add text annotations for Q-values
    for i, (bar, q) in enumerate(zip(bars, q_vals)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{q:.2f}', ha='center', va='bottom' if q > 0 else 'top',
                fontsize=8)
    
    fig.tight_layout()
    return fig

def main():
    args = get_args_and_merge_config()
    args.critic_lr = float(args.critic_lr)
    # COMMENTED OUT: Actor learning rate no longer needed
    # args.actor_lr = float(args.actor_lr)
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

    if "full_scratch" in args.dino_encoder:
        print("training vc1 from scratch")
        for m in shared_wm.modules():
            if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):
                torch.nn.init.xavier_uniform_(m.weight)
        for p in shared_wm.parameters():
            p.requires_grad = True



    train_envs = [RawDubinsEnv(device=device, with_proprio=args.with_proprio) for _ in range(args.training_num)]
    test_envs = [RawDubinsEnv(device=device, with_proprio=args.with_proprio) for _ in range(args.test_num)]

    # Get dimensions
    dummy_obs, _ = train_envs[0].reset()
    state_dim = encode_batch_optimized([dummy_obs], shared_wm, device, args.with_proprio, requires_grad=False).shape[1]
    # COMMENTED OUT: Action dim and max_action no longer needed for discrete actions
    # action_dim = train_envs[0].action_space.shape[0]
    # max_action = torch.tensor(train_envs[0].action_space.high, device=device, dtype=torch.float32)

    # MODIFIED: Initialize critic for DDQN (outputs Q-values for all actions)
    critic = Critic(state_dim, args.num_actions, args.critic_net, args.critic_activation).to(device)

    # COMMENTED OUT: Actor no longer needed
    # actor = Actor(state_dim, action_dim, args.control_net, args.actor_activation, max_action).to(device)
    # actor_optim = torch.optim.AdamW(actor.parameters(), lr=args.actor_lr)

    critic_optim = torch.optim.AdamW(critic.parameters(), lr=args.critic_lr)

    if args.with_finetune:
        encoder_optim = torch.optim.AdamW(shared_wm.parameters(), lr=args.encoder_lr)
    else:
        encoder_optim = None

    # MODIFIED: Initialize DDQN policy instead of DDPG
    policy = AvoidDDQNPolicy(
        critic=critic,
        critic_optim=critic_optim,
        tau=args.tau,
        gamma=args.gamma_pyhj,
        exploration_noise=args.exploration_noise,  # Used as epsilon for epsilon-greedy
        device=device,
        with_proprio=args.with_proprio,
        num_actions=args.num_actions,
        action_space=train_envs[0].action_space
    )
    print("args.with_proprio",args.with_proprio)
    # MODIFIED: Use discrete action dimension (1) for buffer initialization
    buffer = OptimizedReplayBuffer(args.buffer_size, device, dummy_obs, 1)

    # Initialize parallel collector
    collector = ParallelEnvCollector(train_envs, policy, shared_wm, device, args.with_proprio)

    timestamp = datetime.now().strftime("%m%d_%H%M")
    log_dir = Path(f"runs/ddqn_hj_latent/{args.dino_encoder}-{timestamp}/")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    wandb.init(
        project=f"ddqn-hj-latent-dubins",
        name=f"ddqn-{args.dino_encoder}-{timestamp}",
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
        collector.collect_trajectories(buffer,5000)
        
        # Train with progress bar
        pbar = tqdm(range(args.step_per_epoch), desc=f"Epoch {epoch}", unit="step")
        for step in pbar:
            batch = buffer.sample(args.batch_size_pyhj)
            metrics = policy.learn(batch, shared_wm, args.with_finetune, encoder_optim)
            
            pbar.set_postfix({
                # REMOVED: Actor loss (no longer exists)
                "critic_loss": f"{metrics['loss/critic']:.8f}",
                "enc_grad": f"{metrics['grad_norm']:.8f}",
                "buffer": len(buffer)
            })
            wandb.log(metrics, step=epoch * args.step_per_epoch + step)
        pbar.close()
        # Plot HJ using optimized function
        fig1, fig2 = plot_hj_optimized(policy, helper_env, shared_wm, thetas, args, device)
        
        # NEW: Plot best actions grid
        fig_actions = plot_best_actions_grid(policy, helper_env, shared_wm, thetas, args, device)
        
        # NEW: Plot Q-values at specific positions (near obstacles)
        # You can adjust these positions based on where your obstacles are
        test_positions = [
            (1.0, 1.0),   # Near obstacle
            (-1.0,1.0),
            (0.0,0.0),
            (0.5,-2),
            (0.5,1),
            (-0.5,2),
            (2.0, 2.0),   # Goal position
            (0.5, 0.5),   # Starting area
        ]
        
        for pos in test_positions:
            fig_q = plot_all_q_values_comparison(
                policy, helper_env, shared_wm, 
                theta=0.0,  # You can vary this
                position=pos,
                args=args, device=device
            )
            wandb.log({
                f"Q_values_at_{pos[0]}_{pos[1]}": wandb.Image(fig_q)
            })
            plt.close(fig_q)
        
        wandb.log({
            "HJ_latent/binary": wandb.Image(fig1),
            "HJ_latent/continuous": wandb.Image(fig2),
            "HJ_latent/best_actions": wandb.Image(fig_actions),  # NEW
        })
        plt.close(fig1)
        plt.close(fig2)
        plt.close(fig_actions)

        # Save checkpoints
        epoch_dir = log_dir / f"epoch_{epoch}"
        epoch_dir.mkdir(exist_ok=True)
        # COMMENTED OUT: Actor checkpoint no longer needed
        # torch.save(policy.actor.state_dict(), epoch_dir / "actor.pth")
        torch.save(policy.critic.state_dict(), epoch_dir / "critic.pth")
        # if args.with_finetune :
        #     # FIXED: Properly unwrap models before saving (matching train.py pattern)
        #     def unwrap_model_if_needed(model):
        #         """Unwrap model from DDP/accelerator wrapper if present"""
        #         if hasattr(model, "module"):
        #             return model.module  # Unwrap DDP
        #         return model
            
        #     ckpt = {
        #         # Unwrap each component before saving
        #         "encoder":         unwrap_model_if_needed(shared_wm.encoder),
        #         "predictor":       unwrap_model_if_needed(shared_wm.predictor),
        #         "proprio_encoder": unwrap_model_if_needed(shared_wm.proprio_encoder),
        #         "action_encoder":  unwrap_model_if_needed(shared_wm.action_encoder),
        #         # Handle decoder which might be None
        #         "decoder":         unwrap_model_if_needed(shared_wm.decoder) if shared_wm.decoder is not None else None,
        #         # Also save the epoch
        #         "epoch":           epoch,
        #     }
        #     torch.save(ckpt, epoch_dir / "model_latest.pth")
        if args.with_finetune and epoch%20==0:
            torch.save(shared_wm.state_dict(), epoch_dir / "wm.pth")



    print("Training complete.")

if __name__ == "__main__":
    main()
    
    
''' 
NOTE
currently loss to encoder backpropped only through the critic not also through the actor
MODIFIED: Now using DDQN with discrete action space instead of DDPG with continuous actions
'''


# python train_HJ_dubinslatent_withfinetune_ddqn.py --dino_ckpt_dir "checkpoints/wm/dubins"  --config train_HJ_configs.yaml --dino_encoder dino_cls  --nx 50 --ny 50 --step-per-epoch 200 --total-episodes 200 --batch_size-pyhj 64 --gamma-pyhj 0.99 --actor-gradient-steps 2 --critic_net 512 512 512 --control_net 512 512 512



# python train_HJ_dubinslatent_withfinetune_ddqn.py --dino_ckpt_dir "checkpoints/wm/dubins"  --config train_HJ_configs.yaml --dino_encoder dino_cls --nx 50 --ny 50 --step-per-epoch 200 --total-episodes 100 --batch_size-pyhj 64 --gamma-pyhj 0.99 --actor-gradient-steps 2 --critic-net 128 128 128  --control-net 128 128 128  --with_finetune --encoder_lr 1e-6



# -1.0 (Red) = Turn hard left
# 0.0 (White) = Go straight
# +1.0 (Blue) = Turn hard right


# Right Subplot - "Q-value Difference" Colorbar:
# This shows the confidence of the action selection: Q(best_action) - Q(second_best_action)

# High values (yellow) = Large gap between best and second-best action → Network is confident
# Low values (dark purple) = Small gap → Network thinks multiple actions are similarly good/bad
# Near zero = Network can't distinguish between actions