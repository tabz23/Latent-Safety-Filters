import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import time

def visualize_dubins_trajectories(
    data_dir="datasets/dubins1800_withcost",
    num_traj=2,             # Number of trajectories to visualize
    # show_steps=40,          # Number of steps per trajectory to show
    fps=15,                 # Animation speed
    save_dir="env/dubins"
):
    obs_dir = Path(data_dir) / "obses"
    actions = torch.load(Path(data_dir) / "actions.pth")
    states = torch.load(Path(data_dir) / "states.pth")
    costs = torch.load(Path(data_dir) / "costs.pth")  # ADDED: load costs
    seq_lengths = torch.load(Path(data_dir) / "seq_lengths.pth")

    # Create save directory
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loaded {len(seq_lengths)} trajectories.")
    print(f"seq_lengths.shape: {seq_lengths.shape}")
    print(f"actions type: {type(actions)}, length: {len(actions)}")
    print(f"states type: {type(states)}, length: {len(states)}")
    print(f"costs type: {type(costs)}, length: {len(costs)}")  # ADDED: print costs info
    print(f"actions[0] shape: {actions[0].shape}")  # first trajectory
    print(f"states[0] shape: {states[0].shape}")
    print(f"costs[0] shape: {costs[0].shape}")  # ADDED: print costs shape
    
    n_vis = min(num_traj, len(seq_lengths))
    idxs = np.random.choice(len(seq_lengths), size=n_vis, replace=False)

    # Counter for saved images
    saved_count = 0
    max_save = 20  # Save up to 20 example images
    saved_images_info = []  # Track saved images and their costs

    for i, idx in enumerate(idxs):
        print(f"\nTrajectory {idx}: Length = {seq_lengths[idx].item()}")
        traj_states = states[idx][:seq_lengths[idx]]
        traj_costs = costs[idx][:seq_lengths[idx]]  # ADDED: get trajectory costs
        
        print("  State [first]:", traj_states[0].numpy())
        print("  State [last]:", traj_states[-1].numpy())
        print(f"  Costs [first 10]: {traj_costs[:10].numpy()}")  # ADDED: print first 10 costs
        print(f"  Costs [last 10]: {traj_costs[-10:].numpy()}")  # ADDED: print last 10 costs
        print(f"  Total unsafe steps: {traj_costs.sum().item()}/{len(traj_costs)}")  # ADDED: count unsafe steps

        imgs = torch.load(obs_dir / f"episode_{idx:03d}.pth")  # [T, H, W, C] uint8
        T = imgs.shape[0]
        print(f"  Images shape: {imgs.shape}")
        
        fig, ax = plt.subplots(figsize=(4, 4))
        plt.title(f"Dubins Trajectory {idx}")

        for t in range(T):
            ax.clear()
            ax.imshow(imgs[t].numpy())
            ax.set_axis_off()
            cost_val = traj_costs[t].item()
            safety_text = "UNSAFE" if cost_val > 0.5 else "SAFE"
            ax.set_title(f"Traj {idx} - Step {t+1}/{T} - {safety_text} (cost: {cost_val})")
            
            # Save some example images with their costs
            if saved_count < max_save and t % 2 == 0:  # Save every 5th step
                img_save_path = save_dir / f"traj_{idx}_step_{t}_cost_{cost_val:.0f}.png"
                plt.savefig(img_save_path, bbox_inches='tight', dpi=150)
                saved_images_info.append({
                    'path': img_save_path,
                    'trajectory': idx,
                    'step': t,
                    'cost': cost_val,
                    'safety': safety_text
                })
                saved_count += 1
            
            plt.pause(1.0 / fps)
        plt.show(block=False)
        time.sleep(1)
        plt.close(fig)

    # Print summary of saved images
    print(f"\n--- Saved Images Summary ---")
    print(f"Total images saved: {len(saved_images_info)}")
    if saved_images_info:
        safe_count = sum(1 for img in saved_images_info if img['cost'] <= 0.5)
        unsafe_count = len(saved_images_info) - safe_count
        print(f"Safe images: {safe_count}, Unsafe images: {unsafe_count}")
        print("Saved image details:")
        for img_info in saved_images_info:
            print(f"  {img_info['path'].name} - Traj {img_info['trajectory']}, Step {img_info['step']}, {img_info['safety']} (cost: {img_info['cost']})")
    else:
        print("No images were saved.")

    # Save a summary of costs across all trajectories
    all_costs = torch.cat(costs)
    cost_summary = {
        'total_steps': len(all_costs),
        'unsafe_steps': (all_costs > 0.5).sum().item(),
        'safe_steps': (all_costs <= 0.5).sum().item(),
        'safety_rate': (all_costs <= 0.5).float().mean().item()
    }
    
    print(f"\n--- Cost Summary ---")
    print(f"Total steps across all trajectories: {cost_summary['total_steps']}")
    print(f"Unsafe steps: {cost_summary['unsafe_steps']}")
    print(f"Safe steps: {cost_summary['safe_steps']}")
    print(f"Safety rate: {cost_summary['safety_rate']:.3f}")
    
    # Save cost summary to file
    summary_path = save_dir / "cost_summary.txt"
    with open(summary_path, 'w') as f:
        for key, value in cost_summary.items():
            f.write(f"{key}: {value}\n")
    print(f"Saved cost summary to: {summary_path}")

if __name__ == "__main__":
    visualize_dubins_trajectories()