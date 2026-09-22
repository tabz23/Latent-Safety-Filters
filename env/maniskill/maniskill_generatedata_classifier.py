


import mani_skill.envs
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import argparse
from pathlib import Path
#from Controller import CheckpointController
import os
import numpy as np
import os
os.environ['MUJOCO_GL'] = 'egl'  # or 'osmesa' or 'egl'
os.environ['PYOPENGL_PLATFORM'] = 'egl'
import gymnasium as gym

class CheckpointController:
    """
    A simple class to load and use trained PPO checkpoints for data recording
    
    Usage:
        controller = CheckpointController("path/to/checkpoint.pt", env)
        action = controller.get_action(observation)
    """
    
    def __init__(self, checkpoint_path, env, device="auto", deterministic=True):
        """
        Initialize the controller with a trained checkpoint
        
        Args:
            checkpoint_path (str): Path to the .pt checkpoint file
            env: ManiSkill environment (used to get action/observation spaces)
            device (str): Device to run on ("auto", "cpu", "cuda")
            deterministic (bool): Whether to use deterministic actions by default
        """
        self.checkpoint_path = checkpoint_path
        self.deterministic = deterministic
        
        # Setup device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        #print(f"CheckpointController using device: {self.device}")
        
        # Store environment info
        self.obs_space = env.single_observation_space
        self.action_space = env.single_action_space
        
        # Setup action clipping
        self.action_low = torch.from_numpy(self.action_space.low).to(self.device)
        self.action_high = torch.from_numpy(self.action_space.high).to(self.device)
        
        # Initialize and load the agent
        self.agent = self._create_agent()
        self._load_checkpoint()
        
        #print(f"✅ Loaded checkpoint: {os.path.basename(checkpoint_path)}")
    
    def _layer_init(self, layer, std=np.sqrt(2), bias_const=0.0):
        """Initialize network layers"""
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
        return layer
    
    def _create_agent(self):
        """Create the agent with the same architecture as training"""
        
        class Agent(nn.Module):
            def __init__(self, obs_dim, action_dim, layer_init_fn):
                super().__init__()
                self.critic = nn.Sequential(
                    layer_init_fn(nn.Linear(obs_dim, 256)),
                    nn.Tanh(),
                    layer_init_fn(nn.Linear(256, 256)),
                    nn.Tanh(),
                    layer_init_fn(nn.Linear(256, 256)),
                    nn.Tanh(),
                    layer_init_fn(nn.Linear(256, 1)),
                )
                self.actor_mean = nn.Sequential(
                    layer_init_fn(nn.Linear(obs_dim, 256)),
                    nn.Tanh(),
                    layer_init_fn(nn.Linear(256, 256)),
                    nn.Tanh(),
                    layer_init_fn(nn.Linear(256, 256)),
                    nn.Tanh(),
                    layer_init_fn(nn.Linear(256, action_dim), std=0.01*np.sqrt(2)),
                )
                self.actor_logstd = nn.Parameter(torch.ones(1, action_dim) * -0.5)

            def get_action(self, x, deterministic=False):
                action_mean = self.actor_mean(x)
                if deterministic:
                    return action_mean
                action_logstd = self.actor_logstd.expand_as(action_mean)
                action_std = torch.exp(action_logstd)
                probs = Normal(action_mean, action_std)
                return probs.sample()
        
        obs_dim = np.array(self.obs_space.shape).prod()
        action_dim = np.prod(self.action_space.shape)
        
        agent = Agent(obs_dim, action_dim, self._layer_init).to(self.device)
        return agent
    
    def _load_checkpoint(self):
        """Load the checkpoint weights"""
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")
        
        checkpoint_data = torch.load(self.checkpoint_path, map_location=self.device)
        self.agent.load_state_dict(checkpoint_data)
        self.agent.eval()
    
    def _clip_action(self, action):
        """Clip actions to valid range"""
        return torch.clamp(action, self.action_low, self.action_high)
    
    def _prepare_observation(self, obs):
        """Convert observation to tensor and move to device"""
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs).float()
        elif not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs).float()
        
        obs = obs.to(self.device)
        
        # Add batch dimension if needed
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
            return obs, True  # True means we added batch dim
        
        return obs, False
    
    def get_action(self, observation, deterministic=None):
        """
        Get action from the trained policy
        
        Args:
            observation: Environment observation (numpy array or torch tensor)
            deterministic (bool): Override default deterministic setting
        
        Returns:
            action: Action ready for env.step() (numpy array)
        """
        if deterministic is None:
            deterministic = self.deterministic
        
        # Prepare observation
        obs, added_batch = self._prepare_observation(observation)
        
        # Get action from agent
        with torch.no_grad():
            action = self.agent.get_action(obs, deterministic=deterministic)
            action = self._clip_action(action)
        
        # Remove batch dimension if we added it
        if added_batch:
            action = action.squeeze(0)
        
        # Convert to numpy for environment
        return action.squeeze(1).squeeze(0).cpu().numpy()
    
    def get_action_tensor(self, observation, deterministic=None):
        """
        Get action as torch tensor (useful if you need tensor operations)
        
        Args:
            observation: Environment observation
            deterministic (bool): Override default deterministic setting
        
        Returns:
            action: Action as torch tensor
        """
        if deterministic is None:
            deterministic = self.deterministic
        
        obs, added_batch = self._prepare_observation(observation)
        
        with torch.no_grad():
            action = self.agent.get_action(obs, deterministic=deterministic)
            action = self._clip_action(action)
        
        if added_batch:
            action = action.squeeze(0)
        
        return action.squeeze(1).squeeze(0)
    
    def set_deterministic(self, deterministic):
        """Change whether to use deterministic actions"""
        self.deterministic = deterministic
    
    def get_stats(self):
        """Get information about the controller"""
        return {
            "checkpoint_path": self.checkpoint_path,
            "device": str(self.device),
            "deterministic": self.deterministic,
            "obs_shape": self.obs_space.shape,
            "action_shape": self.action_space.shape,
            "action_range": [self.action_space.low.min(), self.action_space.high.max()]
        }
    
    def __repr__(self):
        return f"CheckpointController(checkpoint='{os.path.basename(self.checkpoint_path)}', device='{self.device}')"


def collect_episodes(name, num_episodes, max_steps, obs_mode, output_dir,
                     expert_ckpt="checkpoints/maniskill_ppo_expert.pt",
                     expert_ckpt_2="checkpoints/maniskill_ppo_expert_2.pt"):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    obses_dir = output_path / "obses"
    obses_dir.mkdir(exist_ok=True)
    all_actions = []
    all_states = []
    all_costs = []
    seq_lengths = []

    checkpoint_path = expert_ckpt
    checkpoint_2_path = expert_ckpt_2
    halfway = num_episodes // 2

    for episode_idx in range(num_episodes):
        print(f"Episode {episode_idx + 1}/{num_episodes}")

        # Initalize environment
        env = gym.make(name, obs_mode=obs_mode, sensor_configs=dict(width=224, height=224))
        
        # Use different controller depending on which half of the data we are in
        ckpt = checkpoint_path if episode_idx < halfway else checkpoint_2_path
        controller = CheckpointController(ckpt, env, deterministic=True)

        episode_actions = []
        episode_states = []
        episode_costs = []
        episode_obs = []
        obs, _ = env.reset()

        for step in range(max_steps):
            state = env.unwrapped.get_state()
            if len(state.shape) > 1:
                state = state.squeeze(0)
            episode_states.append(state)
            image = env.unwrapped.render_sensors().squeeze(0)[:, -224:, :]
            episode_obs.append(image)

            action = controller.get_action(obs)
            episode_actions.append(torch.tensor(action, dtype=torch.float32).squeeze())
            obs, _, terminated, truncated, _ = env.step(action)
            cost = calculate_cost(env)
            episode_costs.append(torch.tensor(cost, dtype=torch.float32))

            if terminated or truncated:
                break

        env.close()

        all_actions.append(torch.stack(episode_actions))
        all_states.append(torch.stack(episode_states))
        all_costs.append(torch.stack(episode_costs))
        seq_lengths.append(len(episode_actions))

        torch.save(torch.stack(episode_obs).cpu(), obses_dir / f"episode_{episode_idx}.pth")

    # Pad and save
    max_len = max(seq_lengths)
    padded_actions = torch.zeros(num_episodes, max_len, all_actions[0].shape[-1])
    padded_states = torch.zeros(num_episodes, max_len, all_states[0].shape[-1])
    padded_costs = torch.zeros(num_episodes, max_len)

    for i, (actions, states, costs) in enumerate(zip(all_actions, all_states, all_costs)):
        length = len(actions)
        padded_actions[i, :length] = actions
        padded_states[i, :length] = states
        padded_costs[i, :length] = costs

    torch.save(padded_actions, output_path / "actions.pth")
    torch.save(padded_states, output_path / "states.pth")
    torch.save(torch.tensor(seq_lengths), output_path / "seq_lengths.pth")
    torch.save(padded_costs, output_path / "costs.pth")

    print(f"Saved {num_episodes} episodes to {output_path}")


def calculate_cost(env, collision_threshold = 1e-6):
    #Get objects from environment
    bowl = env.unwrapped.bowl
    scene = env.unwrapped.scene
    robot = env.unwrapped.agent.robot
    #Find the correct hand link
    all_links = robot.get_links()
    right_hand_link = next((link for link in all_links if link.name == 'right_palm_link'), None)
    #Calculate contact forces
    contact_forces = scene.get_pairwise_contact_forces(right_hand_link, bowl)
    if contact_forces is not None and len(contact_forces) > 0:
        forces_magnitudes = torch.norm(contact_forces, dim = -1)
        total_force = torch.sum(forces_magnitudes).item()
         #Returns 1.0 if there is a collision
        if total_force > collision_threshold:
            return 1.0
    return 0.0

def main():
    #Added argparser so that all variables can be changed easily 
    parser = argparse.ArgumentParser(description= ' Record Maniskill')
    parser.add_argument('--name', type= str, default= "UnitreeG1PlaceAppleInBowl-v1", help = 'Name of environment')
    parser.add_argument('--num-episodes', type = int, default = 3000)
    parser.add_argument('--max-steps', type = int, default = 100, help = 'Number of steps')
    parser.add_argument('--obs-mode',type = str, default = "state", help = 'Observation mode')
    parser.add_argument('--output-dir', type = str, default = 'datasets/maniskill3000classif')
    parser.add_argument('--expert-ckpt', type = str, default = 'checkpoints/maniskill_ppo_expert.pt', help = 'Path to the expert PPO policy checkpoint')
    parser.add_argument('--expert-ckpt-2', type = str, default = 'checkpoints/maniskill_ppo_expert_2.pt', help = 'Path to the second expert PPO policy checkpoint (used for the second half of episodes)')
    args = parser.parse_args()

    collect_episodes(args.name, args.num_episodes, args.max_steps, args.obs_mode, args.output_dir, args.expert_ckpt, args.expert_ckpt_2)

if __name__ == "__main__":
    main()

