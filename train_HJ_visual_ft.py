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
from env.cargoal.CarGoal import CarGoal
from datasets.img_transforms import default_transform
from einops import rearrange
from utils import load_dreamer
# Load DINO-WM via plan.load_model
from plan import load_model

# Underlying Dubins Gym env
from gymnasium.spaces import Box

# Set up matplotlib config
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
os.environ['MUJOCO_GL'] = 'osmesa'
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
        default="checkpoints/wm/cargoal",
        help="Where to find the DINO-WM checkpoints"
    )
    parser.add_argument(
        "--output_dir", type=str, default="outputs/hj_debug",
        help="Directory to save debug plots"
    )
    parser.add_argument(
        "--config", type=str, default="train_HJ_visual_configs.yaml",
        help="Path to your flat YAML of hyperparameters"
    )
    parser.add_argument(
        "--with_proprio", action="store_true",
        help="Flag to include proprioceptive information in latent encoding"
    )
    parser.add_argument(
        "--dino_encoder", type=str, default="dino_cls",
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

class RawCarGoalEnv(gym.Env):
    def __init__(self, device: str = None, with_proprio: bool = False):
        super().__init__()
        self.env = CarGoal()
        self.device = torch.device(device) if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.with_proprio = with_proprio
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def reset(self, state=None):
        reset_out = self.env.reset()
        frame = self.env._env.task.render(128, 128, mode="rgb_array", camera_name="vision", cost={})
        # Gym reset returns obs; if obs is tuple unpack
        obs = {
            'proprio': reset_out["vector"][:24],
            'visual': frame
        }
        obs_proc = {k: np.expand_dims(np.array(v), axis=0) for k, v in reset_out.items()}
        return obs, {"input_dreamer":obs_proc}

    def step(self, action):
        obs_raw, cost, terminated, info = self.env.step(action)
        frame = self.env._env.task.render(128, 128, mode="rgb_array", camera_name="vision", cost={})
        truncated = False
        # extract obs if tuple
        obs = {
            'proprio': obs_raw["vector"][:24],
            'visual': frame
        }
        obs_proc = {k: np.expand_dims(np.array(v), axis=0) for k, v in obs_raw.items()}
        info["input_dreamer"] = obs_proc
        # override reward with safety metric
        # h_s = cost if cost >= 0 else 5*cost ##I multiplied by 3 to make HJ easier to learn
        h_s = 10*cost ##I multiplied by 3 to make HJ easier to learn
        return obs, h_s, terminated, truncated, info

class OptimizedReplayBuffer:
    """Optimized replay buffer using pre-allocated tensors"""
    def __init__(self, size: int, device: str, obs_example, action_dim):
        self.size = size
        self.device = torch.device(device)
        self.position = 0
        self.filled = 0
        self.transform = default_transform()
        # Extract dimensions from example observation
        if isinstance(obs_example, dict):
            visual_shape = obs_example['visual'].shape
            proprio_dim = obs_example['proprio'].shape[0]
        else:
            visual_shape = obs_example[0].shape
            proprio_dim = obs_example[1].shape[0]
        
        # Pre-allocate tensors on CPU to save GPU memory
        # h, w, c = visual_shape
        h, w, c = 224,224,3
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

        if isinstance(visual, torch.Tensor):
            visual = visual.numpy()
            visual_next = visual_next.numpy()

        if visual.max() > 1:
            visual = visual / 255.0
            visual_next = visual_next / 255.0
        visual = rearrange(visual, "H W C -> C H W")
        visual = self.transform(torch.from_numpy(visual))
        visual_next = rearrange(visual_next, "H W C -> C H W")
        visual_next = self.transform(torch.from_numpy(visual_next))
        # Convert and store directly in pre-allocated tensors
        self.visual_buffer[self.position] = visual.float()
        self.proprio_buffer[self.position] = torch.from_numpy(proprio).float()
        self.action_buffer[self.position] = torch.from_numpy(act).float()
        self.reward_buffer[self.position] = rew
        self.next_visual_buffer[self.position] = visual_next.float()
        self.next_proprio_buffer[self.position] = torch.from_numpy(proprio_next).float()
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
            visual_np = np.transpose(visual, (2, 0, 1)) / 255.0
            visual_list.append(torch.from_numpy(visual_np).float())
            proprio_list.append(torch.from_numpy(proprio).float())
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
        self.save_failure_examples = False

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
        if not self.save_failure_examples:
            if rew_batch[rew_batch < 0].shape[0] >= 8:
                idx = (rew_batch < 0).squeeze(1)
                obs_batch_unsafe = (obs_batch[0][idx][:8],obs_batch[1][idx][:8])
                act_batch_unsafe = act_batch[idx][:8]
                rew_batch_unsafe = rew_batch[idx][:8]
                self.eval_batch_unsafe = (obs_batch_unsafe, act_batch_unsafe, rew_batch_unsafe)
                idx = (rew_batch >= 0).squeeze(1)
                obs_batch_safe = (obs_batch[0][idx][:8],obs_batch[1][idx][:8])
                act_batch_safe = act_batch[idx][:8]
                rew_batch_safe = rew_batch[idx][:8]
                self.eval_batch_safe = (obs_batch_safe, act_batch_safe, rew_batch_safe)
                self.save_failure_examples = True
        # Encode current states
        z = encode_batch_optimized(obs_batch, wm, self.device, self.with_proprio, requires_grad=requires_grad)
        
        # Compute target returns using avoid Bellman equation
        target_returns = self.compute_nstep_return(obs_batch, act_batch, rew_batch, obs_next_batch, done_batch, wm)
        
        # Critic update - using original's _mse_optimizer logic
        with torch.no_grad():
            pred_actions = self.actor(z).mean(dim=0)
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
            "action/action_1": pred_actions[0].item(),
            "action/action_2": pred_actions[1].item(),
            "grad_norm": grad_norm if with_finetune else 0.0
        }

    def exploration_noise(self, wm, act, batch_obs):
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
            z = encode_batch_optimized(batch_obs, wm, self.device, self.with_proprio, requires_grad=False)
            rand_act_tensor = torch.from_numpy(rand_act).float().to(self.device)
            
            with torch.no_grad():
                values = self.critic(z, rand_act_tensor).cpu().detach().numpy()
            
      
            # act = np.where(values < 0.0, rand_act, act)
        
        
            #"Where unsafe, keep policy action; where safe, use random action"
            
            # Where random actions would be unsafe (values < 0.0): stick with the policy action (act)
            # Where random actions would be safe (values >= 0.0): use the random action (rand_act)
            act = np.where(values < 0.0, act, rand_act)
            
        # Warmup override - exact as original
        if self.warmup:
            act = np.random.uniform(self.action_low, self.action_high, act.shape)

        return act

    def plot_obs_with_hj_values(self, wm, encoder_name, output_dir):
        print("--------------------------------")
        print("Saving Images for Evaluation")
        obs_batch_unsafe, act_batch_unsafe, rew_batch_unsafe = self.eval_batch_unsafe
        obs_batch_safe, act_batch_safe, rew_batch_safe = self.eval_batch_safe
        rew_batch_safe = rew_batch_safe.squeeze(-1).cpu().numpy()
        rew_batch_unsafe = rew_batch_unsafe.squeeze(-1).cpu().numpy()
        z_safe = encode_batch_optimized(obs_batch_safe, wm, self.device, self.with_proprio, requires_grad=False)
        z_unsafe = encode_batch_optimized(obs_batch_unsafe, wm, self.device, self.with_proprio, requires_grad=False)
        with torch.no_grad():
            hj_values_unsafes = self.critic(z_unsafe, act_batch_unsafe).squeeze(-1).cpu().numpy()
            hj_values_safes = self.critic(z_safe, act_batch_safe).squeeze(-1).cpu().numpy()
        plt.figure(figsize=(25, 10))
        for i in range(8):
            safe_img = np.transpose((obs_batch_safe[0][i]*0.5+0.5).cpu().numpy(), (1, 2, 0))
            unsafe_img = np.transpose((obs_batch_unsafe[0][i]*0.5+0.5).cpu().numpy(), (1, 2, 0))
            plt.subplot(2, 8, i+1)
            plt.imshow(unsafe_img)
            plt.title(f"HJ Value: {hj_values_unsafes[i]:.2f}")
            plt.xlabel(f"Cost: {rew_batch_unsafe[i]:.2f}")
            plt.subplot(2, 8, i+9)
            plt.imshow(safe_img)
            plt.title(f"HJ: {hj_values_safes[i]:.2f}")
            plt.xlabel(f"Cost: {rew_batch_safe[i]:.2f}")
        os.makedirs(output_dir, exist_ok=True)
        plot_path = os.path.join(output_dir, f"obs_with_hj_values_{encoder_name}.png")
        plt.savefig(plot_path)
        wandb.log({"visual/obs_with_hj_values": wandb.Image(plot_path)})
        plt.close()
        print("--------------------------------")


            

class ParallelEnvCollector:
    """Collects transitions from multiple envs in parallel with safety switching"""
    def __init__(self, envs, dreamer_agent, shared_wm, with_proprio, policy, device):
        self.envs = envs
        self.policy = policy
        self.device = device
        self.dreamer_agent = dreamer_agent
        self.shared_wm = shared_wm
        self.with_proprio = with_proprio
        self.num_envs = len(envs)

    def collect_trajectories(self, buffer, max_steps: int=1000, warm_up=False):
        """Collect transitions using smart controller with HJ safety switching"""
        reset_out = [env.reset() for env in self.envs]
        obs_list = [out[0] for out in reset_out]
        info_list = [out[1] for out in reset_out]
        done_flags = [False] * self.num_envs
        steps = 0

        while steps < max_steps:
            active_idxs = [i for i, d in enumerate(done_flags) if not d]
            if not active_idxs:
                for i in range(self.num_envs):
                    reset_out = self.envs[i].reset()
                    obs_list[i] = reset_out[0]
                    info_list[i] = reset_out[1]
                    done_flags[i] = False
                continue

           # 3) batch‑encode only active observations
            active_info = [info_list[i] for i in active_idxs]
            active_info_dreamer = [info["input_dreamer"] for info in active_info]
            dreamer_input = {}
            for k in active_info_dreamer[0].keys():
                dreamer_input[k] = np.concatenate([info[k] for info in active_info_dreamer],axis=0)
            action_dict, agent_state, _ = self.dreamer_agent(dreamer_input, None)
            actions = action_dict["action"]
            if not warm_up:
                active_obs = [obs_list[i] for i in active_idxs]
                z = encode_batch_optimized(active_obs,
                                        self.shared_wm,
                                        self.device,
                                        self.with_proprio,
                                        requires_grad=False)
                q_values = self.policy.critic(z, actions).detach().cpu().numpy()
                with torch.no_grad():
                    safe_actions = self.policy.actor(z).cpu().numpy()
                raw_actions = np.where(q_values < 0.0, safe_actions, actions.cpu().numpy())
                actions = self.policy.exploration_noise(self.shared_wm, raw_actions, active_obs)
            else:
                actions = actions.cpu().numpy()

            # 5) Apply exact same exploration as original
            # Note: pass active_obs as the batch argument
                
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
                info_list[env_idx] = info
                done_flags[env_idx] = done

                if steps >= max_steps:
                    break


    def collect_trajectories_backup(self, buffer, max_steps: int=1000, use_dreamer=False):
        """Collect transitions using smart controller with HJ safety switching"""
        reset_out = [env.reset() for env in self.envs]
        obs_list = [out[0] for out in reset_out]
        info_list = [out[1] for out in reset_out]
        done_flags = [False] * self.num_envs
        steps = 0

        while steps < max_steps:
            active_idxs = [i for i, d in enumerate(done_flags) if not d]
            if not active_idxs:
                for i in range(self.num_envs):
                    reset_out = self.envs[i].reset()
                    obs_list[i] = reset_out[0]
                    info_list[i] = reset_out[1]
                    done_flags[i] = False
                continue

           # 3) batch‑encode only active observations
            if use_dreamer:
                active_info = [info_list[i] for i in active_idxs]
                active_info_dreamer = [info["input_dreamer"] for info in active_info]
                dreamer_input = {}
                for k in active_info_dreamer[0].keys():
                    dreamer_input[k] = np.concatenate([info[k] for info in active_info_dreamer],axis=0)
                action_dict, agent_state, _ = self.dreamer_agent(dreamer_input, None)
                actions = action_dict["action"].cpu().numpy()
            else:
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
                actions = self.policy.exploration_noise(self.shared_wm, raw_actions, active_obs)
                
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
                info_list[env_idx] = info
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
        args.dino_ckpt_dir = os.path.join(args.dino_ckpt_dir, "dino_cls")
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

    train_envs = [RawCarGoalEnv(device=device, with_proprio=args.with_proprio) for _ in range(args.training_num)]
    test_envs = [RawCarGoalEnv(device=device, with_proprio=args.with_proprio) for _ in range(args.test_num)]

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

    # Use optimized replay buffer
    buffer = OptimizedReplayBuffer(args.buffer_size, device, dummy_obs, action_dim)

    # Initialize parallel collector
    deamer_agent = load_dreamer()
    collector = ParallelEnvCollector(train_envs, deamer_agent, shared_wm, args.with_proprio, policy, device)

    timestamp = datetime.now().strftime("%m%d_%H%M")
    if args.with_finetune:
        args.dino_encoder = args.dino_encoder + "_ft"
    log_dir = Path(f"runs/ddpg_hj_latent_cargoal/{args.dino_encoder}-{timestamp}/")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    wandb.init(
        project=f"ddpg-hj-latent-cargoal",
        name=f"ddpg-{args.dino_encoder}-{timestamp}",
        config=vars(args)
    )
    writer = SummaryWriter(log_dir=log_dir / "logs")


    # Warm up buffer more efficiently
    # print(f"Warming up replay buffer (need ≥ {args.batch_size_pyhj} samples)...")
    # while len(buffer) < 2500:
    #     collector.collect_trajectories(buffer, use_dreamer=True)
    #     print(f"Replay buffer: {len(buffer)} dreamer samples.")
    while len(buffer) < 5000:
        collector.collect_trajectories(buffer, warm_up=True)
        print(f"Replay buffer: {len(buffer)} samples.")
    print(f"Replay buffer: {len(buffer)} samples.")

    for epoch in range(1, args.total_episodes + 1):
        print(f"\n=== Epoch {epoch}/{args.total_episodes} ===")
        
        # Collect data in parallel
        collector.collect_trajectories(buffer, max_steps=2000)
        # if args.dreamer_during_training:
        #     collector.collect_trajectories(buffer, use_dreamer=True)
        
        # Train with progress bar
        pbar = tqdm(range(args.step_per_epoch), desc=f"Epoch {epoch}", unit="step")
        for step in pbar:
            batch = buffer.sample(args.batch_size_pyhj)
            metrics = policy.learn(batch, shared_wm, args.with_finetune, encoder_optim)
            
            pbar.set_postfix({
                "actor_loss": f"{metrics['loss/actor']:.8f}",
                "critic_loss": f"{metrics['loss/critic']:.8f}",
                "a1": f"{metrics['action/action_1']:.8f}",
                "a2": f"{metrics['action/action_2']:.8f}",
                "enc_grad": f"{metrics['grad_norm']:.8f}",
                "buffer": len(buffer)
            })
            wandb.log(metrics, step=epoch * args.step_per_epoch + step)
            if (step +1) % 100 == 0:
                if policy.save_failure_examples:
                    policy.plot_obs_with_hj_values(shared_wm, args.dino_encoder, args.output_dir)
        pbar.close()


        # Save checkpoints
        epoch_dir = log_dir / f"epoch_{epoch}"
        epoch_dir.mkdir(exist_ok=True)
        torch.save(policy.actor.state_dict(), epoch_dir / "actor.pth")
        torch.save(policy.critic.state_dict(), epoch_dir / "critic.pth")
        torch.save(shared_wm.state_dict(), epoch_dir / "wm.pth")

    print("Training complete.")

if __name__ == "__main__":
    main()
    
    