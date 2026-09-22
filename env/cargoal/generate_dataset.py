"""Collect a CarGoal dataset by rolling out the vendored Dreamer nominal
policy (nominal_controller/reaching.pt) in the SafetyCarGoal2Vision-v0 env.

Saves states.pth / actions.pth / costs.pth / seq_lengths.pth +
obses/episode_XXX.pth in the layout expected by datasets/cargoal_dset.py.

Usage:
    python env/cargoal/generate_dataset.py --num_episodes 2000 \
        --output_dir datasets/cargoalnewshort
"""
import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch
from tqdm import trange

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root
from env.cargoal.CarGoal import CarGoal
from utils import load_dreamer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="datasets/cargoalnewshort")
    parser.add_argument("--num_episodes", type=int, default=2000)
    parser.add_argument("--max_steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    save_dir = Path(args.output_dir)
    (save_dir / "obses").mkdir(parents=True, exist_ok=True)

    agent = load_dreamer()
    env = CarGoal(size=(224, 224), seed=args.seed)

    all_actions, all_states, all_costs, seq_lengths = [], [], [], []
    for ep in trange(args.num_episodes, desc="Collecting CarGoal episodes"):
        obs_raw = env.reset()
        obs_proc = {k: np.expand_dims(np.array(v), axis=0) for k, v in obs_raw.items()}
        agent_state = None
        ep_obs, ep_actions, ep_states, ep_costs = [], [], [], []
        for _ in range(args.max_steps):
            with torch.no_grad():
                action_dict, agent_state, _ = agent(obs_proc, agent_state)
            action = action_dict["action"].cpu().numpy()[0]

            obs_raw, cost, done, info = env.step(action)
            obs_proc = {k: np.expand_dims(np.array(v), axis=0) for k, v in obs_raw.items()}

            ep_obs.append(env.render())
            ep_actions.append(action)
            ep_states.append(obs_raw["vector"])
            ep_costs.append(cost)
            if done:
                break

        ep_idx = len(all_actions)
        ep_obs_np = np.stack(ep_obs).astype(np.uint8)  # [T, 224, 224, 3]
        torch.save(torch.tensor(ep_obs_np), save_dir / "obses" / f"episode_{ep_idx:03d}.pth")

        all_actions.append(torch.tensor(np.stack(ep_actions), dtype=torch.float32))
        all_states.append(torch.tensor(np.stack(ep_states), dtype=torch.float32))
        all_costs.append(torch.tensor(np.array(ep_costs), dtype=torch.float32))
        seq_lengths.append(len(ep_obs_np))

    env.close()
    torch.save(all_actions, save_dir / "actions.pth")
    torch.save(all_states, save_dir / "states.pth")
    torch.save(all_costs, save_dir / "costs.pth")
    torch.save(torch.tensor(seq_lengths, dtype=torch.long), save_dir / "seq_lengths.pth")
    print(f"Saved {len(all_actions)} episodes to {save_dir}")


if __name__ == "__main__":
    main()
