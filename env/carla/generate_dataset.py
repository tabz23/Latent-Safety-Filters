import numpy as np
import os
import gym
import car_dreamer
import torch
from gym.envs.registration import register
from pathlib import Path
from tqdm import trange


def collect_carla_dataset(
    outdir="datasets/carla",
    n_traj=5000,
    seed=42
):
    np.random.seed(seed)
    os.makedirs(outdir, exist_ok=True)
    obs_dir = Path(outdir) / "obses"
    obs_dir.mkdir(parents=True, exist_ok=True)
    actions_list = []
    states_list = []
    seq_lengths = []
    costs_list = []

    config = car_dreamer.load_task_configs('carla_follow')
    register(
        id='CarlaFollowWrapperEnv-v0',
        entry_point='carla_wrapper:CarlaWrapper',
    )
    env = gym.make('CarlaFollowWrapperEnv-v0', config=config.env)

    for idx in trange(n_traj, desc="Collecting Carla Trajectories"):
        obs, full_state = env.reset()
        visuals = []
        actions = []
        states = []
        costs = []
        done = False
        t = 0
        while not done:
            # random action
            action = env.compute_continuous_action() 
            obs, reward, done, info = env.step(action)
            visuals.append(obs['visual'])
            actions.append(action)
            state = env.state()
            states.append(state)
            cost = np.linalg.norm(np.array(state[0:2]) - np.array(state[6:8]))
            costs.append(cost)
            t += 1
        # Save trajectory visuals (images)
        visuals = np.stack(visuals).astype(np.uint8)  # [T, H, W, C]
        torch.save(torch.from_numpy(visuals), obs_dir / f"episode_{idx:03d}.pth")
        # Save actions and states as tensors (not stacking across trajectories)
        actions_list.append(torch.tensor(np.stack(actions), dtype=torch.float32))
        states_list.append(torch.tensor(np.stack(states), dtype=torch.float32))
        seq_lengths.append(t)
        costs_list.append(torch.tensor(np.stack(costs), dtype=torch.float32))

    # Save lists of tensors (not stacked!)
    torch.save(actions_list, Path(outdir) / "actions.pth")
    torch.save(states_list, Path(outdir) / "states.pth")
    torch.save(torch.tensor(seq_lengths, dtype=torch.long), Path(outdir) / "seq_lengths.pth")
    torch.save(costs_list, Path(outdir) / "costs.pth")
    print(f"Saved {n_traj} trajectories to {outdir}")

if __name__ == "__main__":
    collect_carla_dataset(outdir='./dataset', n_traj=2000)