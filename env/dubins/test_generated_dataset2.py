# import torch
# # import matplotlib.pyplot as plt
# import numpy as np
# import os
# from pathlib import Path
# import time

# def print_data_structure(name, data):
#     print(f"\n===> {name}:")
#     print(f"  Type: {type(data)}")
#     if isinstance(data, torch.Tensor):
#         print(f"  Shape: {data.shape}")
#         # print(data[1])
#     elif isinstance(data, list):
#         print(f"  List length: {len(data)}")
#         print(f"  First item type: {type(data[0])}")
#         print(f"  First item shape: {data[0].shape if isinstance(data[0], torch.Tensor) else 'N/A'}")
#     else:
#         print("  Unknown data structure.")

# def visualize_dataset_trajectory(
#     data_dir="data/dubins_dataset",
#     num_traj=1,
#     fps=15,
# ):
#     obs_dir = Path(data_dir) / "obses"
#     actions = torch.load(Path(data_dir) / "actions.pth")
#     states = torch.load(Path(data_dir) / "states.pth")
#     seq_lengths = torch.load(Path(data_dir) / "seq_lengths.pth")

#     print(f"\nLoaded {len(seq_lengths)} trajectories from: {data_dir}")
#     print_data_structure("actions", actions)
#     print_data_structure("states", states)

#     n_vis = min(num_traj, len(seq_lengths))
#     idxs = np.random.choice(len(seq_lengths), size=n_vis, replace=False)

#     for i, idx in enumerate(idxs):
#         seq_len = seq_lengths[idx].item()
#         print(f"\nTrajectory {idx}: Length = {seq_len}")
#         traj_actions = actions[idx] if isinstance(actions, list) else actions[idx, :seq_len]
#         traj_states = states[idx] if isinstance(states, list) else states[idx, :seq_len]
#         print("  State [first]:", traj_states[0].numpy())
#         print("  State [last]:", traj_states[-1].numpy())

#         # imgs = torch.load(obs_dir / f"episode_{idx:03d}.pth")  # [T, H, W, C] uint8
#         # T = imgs.shape[0]
#         # fig, ax = plt.subplots(figsize=(4, 4))
#         # plt.title(f"Trajectory {idx}")

#         # for t in range(T):
#         #     ax.clear()
#         #     ax.imshow(imgs[t].numpy())
#         #     ax.set_axis_off()
#         #     ax.set_title(f"Step {t+1}/{T}")
#         #     plt.pause(1.0 / fps)
#         # plt.show(block=False)
#         # time.sleep(1)
#         # plt.close(fig)

#     # print(f"\nobs shape for sample: {imgs.shape}")

# if __name__ == "__main__":
#     # Change to "data/point_maze" to test PointMaze
#     visualize_dataset_trajectory(data_dir="datasets/point_maze")
import torch
import numpy as np
from pathlib import Path

def print_data_structure(name, data):
    print(f"\n===> {name}:")
    print(f"  Type: {type(data)}")
    if isinstance(data, torch.Tensor):
        print(f"  Shape: {data.shape}")
        # show the second element for a quick sanity check
        idx = 1 if data.shape[0] > 1 else 0
        print(f"  Sample [{idx}]:\n{data[idx]}")
    elif isinstance(data, list):
        print(f"  List length: {len(data)}")
        first = data[0]
        print(f"  First item type: {type(first)}")
        if isinstance(first, torch.Tensor):
            print(f"  First item shape: {first.shape}")
    else:
        print("  (no special handling)")

def check_padding(actions, states, seq_lengths):
    max_len = actions.shape[1]
    min_len = int(seq_lengths.min().item())
    print(f"\n=== Padding presence ===")
    if min_len < max_len:
        print(f"Padding detected: shortest trajectory = {min_len}, max = {max_len}")
    else:
        print("No padding: all trajectories have full length")

    print(f"\n=== Length distribution ===")
    uniq, counts = torch.unique(seq_lengths, return_counts=True)
    for length, cnt in zip(uniq.tolist(), counts.tolist()):
        print(f"  Length {length}: {cnt} trajs")

    print(f"\n=== Inspect padded values (actions) ===")
    # mask where t >= true length
    device = seq_lengths.device
    time_idx = torch.arange(max_len, device=device)[None, :]
    mask = time_idx >= seq_lengths[:, None]  # [N, T]
    padded_actions = actions[mask]
    if padded_actions.numel() == 0:
        print("  No padded steps in actions")
    elif torch.all(padded_actions == 0):
        print("  All padded action entries = 0")
    else:
        print(f"  Non-zero padded actions sample:\n{padded_actions.view(-1, actions.shape[-1])[:5]}")

    print(f"\n=== Inspect padded values (states) ===")
    padded_states = states[mask]
    if padded_states.numel() == 0:
        print("  No padded steps in states")
    elif torch.all(padded_states == 0):
        print("  All padded state entries = 0")
    else:
        print(f"  Non-zero padded states sample:\n{padded_states.view(-1, states.shape[-1])[:5]}")

def visualize_dataset_trajectory(
    data_dir="data/dubins_dataset",
    num_traj=1,
):
    data_dir = Path(data_dir)
    actions    = torch.load(data_dir / "actions.pth")    # [N, T, A]
    states     = torch.load(data_dir / "states.pth")     # [N, T, S]
    seq_lengths= torch.load(data_dir / "seq_lengths.pth")# [N]

    print(f"\nLoaded {len(seq_lengths)} trajectories from: {data_dir}")
    print_data_structure("actions", actions)
    print_data_structure("states",  states)
    print_data_structure("seq_lengths", seq_lengths)

    check_padding(actions, states, seq_lengths)

    # --- sample a couple for human inspection ---
    print("\n=== Sample trajectories ===")
    idxs = np.random.choice(len(seq_lengths), size=min(num_traj, len(seq_lengths)), replace=False)
    for idx in idxs:
        L = int(seq_lengths[idx].item())
        print(f" Traj {idx}: true length = {L}")
        print("   first state:", states[idx, 0].numpy())
        print("   last  state:", states[idx, L-1].numpy())
        
    print("seq lengths, ", seq_lengths)

if __name__ == "__main__":
    # e.g. "datasets/point_maze" or your dubins folder
    visualize_dataset_trajectory(data_dir="datasets/dubins1800_withcost",
                                 num_traj=2)
