
# -------------------------------------------------------------------------
import os, cv2, torch, argparse, numpy as np, matplotlib.pyplot as plt
import torch.nn.functional as F
from tqdm import tqdm
from pathlib import Path
from omegaconf import OmegaConf
from matplotlib.colors import LinearSegmentedColormap

# -------------------------------------------------------------------------
# 0.  Project‑specific imports (adjust if you moved code around)
# -------------------------------------------------------------------------
from plan import load_model
from datasets.img_transforms import default_transform     # resize+crop+normalise
from datasets.traj_dset import (
    TrajDataset, TrajSlicerDataset,               # base helpers
    get_train_val_sliced_with_cost, get_train_val_sliced,
    split_traj_datasets
)
from einops import rearrange
from typing import Optional, Callable
# -------------------------------------------------------------------------
# 1.  Utility helpers
# -------------------------------------------------------------------------
"""
Test learned Hamilton-Jacobi safety filter in latent space for Dubins car
"""
import sys
import os
import argparse
from pathlib import Path
from tkinter import N
import torch
import numpy as np
import imageio
from datetime import datetime
from copy import deepcopy
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
import cv2
import random
# Import required modules from your codebase
from plan import load_model
# from env.dubins.dubins import DubinsEnv
from gymnasium.spaces import Box
import gymnasium as gym
from env.cargoal.CarGoal import CarGoal
from utils import load_dreamer
import time
# Set up matplotlib config
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
os.makedirs(os.environ['MPLCONFIGDIR'], exist_ok=True)
os.environ['MUJOCO_GL'] = 'osmesa'


import cv2
import sys
import os
import argparse
from pathlib import Path
import torch
import numpy as np
import imageio
from datetime import datetime
from copy import deepcopy
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
import cv2

# Import required modules from your codebase
from plan import load_model
from env.dubins.dubins import DubinsEnv
from gymnasium.spaces import Box

def smooth_map(smap, ksize=5, sigma=1.0):
    # gaussian blur on the scalar map; keep sign by blurring raw values
    return cv2.GaussianBlur(smap.astype(np.float32), (ksize, ksize), sigma)

# -------------------------------------------------------------------------
# 1.  Utility helpers
# -------------------------------------------------------------------------

def unnormalize(img: torch.Tensor) -> torch.Tensor:
    """Undo `Normalize([0.5],[0.5])`:  [-1,1] ➝ [0,1] (CHW)."""
    return img * 0.5 + 0.5


def to_uint8(img: np.ndarray) -> np.ndarray:
    """Assumes img∈[0,1], returns HWC uint8."""
    return (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)

# -------------------------------------------------------------------------
# 2.  Actor and Critic Networks (I added this now)
# -------------------------------------------------------------------------

class Actor(torch.nn.Module):
    """Actor network - must match training architecture"""
    def __init__(self, state_dim, action_dim, hidden_sizes, activation, max_action):
        super().__init__()
        self.net = self.build_net(state_dim, action_dim, hidden_sizes, activation)
        self.register_buffer('max_action', max_action)
        
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
    """Critic network - must match training architecture"""
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

# -------------------------------------------------------------------------
# 3.  HJ Policy Evaluator (I added this now)
# -------------------------------------------------------------------------


class CarGoalEnvForTesting:
    """Wrapper for DubinsEnv to match the interface expected by HJ code"""
    def __init__(self, device='cuda'):
        self.env = CarGoal()
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.metadata = {"render_fps": 10}  # Add fps for video saving
        
    def reset(self, state=None):
        reset_out = self.env.reset()
        frame = self.env._env.task.render(128, 128, mode="rgb_array", camera_name="vision", cost={})
        # Gym reset returns obs; if obs is tuple unpack
        obs = {
            'proprio': reset_out["vector"][:24],
            'visual': frame
        }
        obs_proc = {k: np.expand_dims(np.array(v), axis=0) for k, v in reset_out.items()}
        return obs, {"input_dreamer":obs_proc, "goal_met": False}
    
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
    
    def render(self, mode='rgb_array'):
        return self.env._env.task.render(224, 224, mode="rgb_array", camera_name="vision", cost={})
    
    def close(self):
        pass

class DubinsEnvForTesting:
    """Wrapper for DubinsEnv to match the interface expected by HJ code"""
    def __init__(self, device='cuda'):
        self.env = DubinsEnv()
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.metadata = {"render_fps": 10}  # Add fps for video saving
        
    def reset(self, state=None):
        if state is not None:
            reset_out = self.env.reset(state=state)
        else:
            reset_out = self.env.reset()
        obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
        return obs, {}
    
    def step(self, action):
        obs_out, reward, done, info = self.env.step(action)
        terminated = done
        truncated = False
        obs = obs_out[0] if isinstance(obs_out, tuple) else obs_out
        h_s = info.get('h', 0.0) * 10  # Multiply by 10 to match training
        return obs, h_s, terminated, truncated, info
    
    def render(self, mode='rgb_array'):
        return self.env.render(mode=mode)
    
    def close(self):
        pass

class HJPolicyEvaluator:
    def __init__(self, actor_path, critic_path, wm, device='cuda', with_proprio=False):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.wm = wm
        self.with_proprio = with_proprio
        
        # Create dummy environment and get dummy observation
        
        dummy_env = DubinsEnvForTesting(device)
        dummy_obs, _ = dummy_env.reset()

        # Convert numpy observation to tensor format for world model
        visual_np = dummy_obs['visual']
        proprio_np = dummy_obs['proprio']
        
        # Convert HWC to CHW if needed
        if visual_np.shape[2] == 3:  # HWC to CHW
            visual_np = np.transpose(visual_np, (2, 0, 1))
        
        # Normalize to world model format [0,1] -> [-1,1] 
        if visual_np.max() > 1.0:
            visual_np = visual_np.astype(np.float32) / 255.0
        visual_np = 2.0 * visual_np - 1.0
        
        # Convert to tensors with proper dimensions
        visual_tensor = torch.from_numpy(visual_np).unsqueeze(0).unsqueeze(0).to(self.device).float()
        proprio_tensor = torch.from_numpy(proprio_np).unsqueeze(0).unsqueeze(0).to(self.device).float()
        
        dummy_obs_tensor = {'visual': visual_tensor, 'proprio': proprio_tensor}
        
        # Now encode the tensor observation
        z = self.encode_observation_tensor(dummy_obs_tensor)
        state_dim = z.shape[1]
        
        action_dim = dummy_env.action_space.shape[0]
        max_action = torch.tensor(dummy_env.action_space.high, device=self.device, dtype=torch.float32)
        
        # Load actor and critic
        self.actor = self._load_actor(actor_path, state_dim, action_dim, max_action)
        self.critic = self._load_critic(critic_path, state_dim, action_dim)
        
        self.actor.eval()
        self.critic.eval()

    def _load_actor(self, path, state_dim, action_dim, max_action):
        """Load actor network (I added this now)"""
        actor = Actor(state_dim, action_dim, [512, 512, 512], 'ReLU', max_action).to(self.device)
        actor.load_state_dict(torch.load(path, map_location=self.device))
        return actor
    
    def _load_critic(self, path, state_dim, action_dim):
        """Load critic network (I added this now)"""
        critic = Critic(state_dim, action_dim, [512, 512, 512], 'ReLU').to(self.device)
        critic.load_state_dict(torch.load(path, map_location=self.device))
        return critic
    
    def encode_observation_tensor(self, obs):
        """Encode observation tensors to latent space (I added this now)"""
        with torch.no_grad():
            lat = self.wm.encode_obs(obs)
        
        # Flatten for HJ networks (I added this now)
        if self.with_proprio:
            z_vis = lat['visual'].reshape(lat['visual'].shape[0], -1)
            z_prop = lat['proprio'].squeeze(1)
            z = torch.cat([z_vis, z_prop], dim=-1)
        else:
            z = lat['visual'].reshape(lat['visual'].shape[0], -1)
        
        return z
    
    def encode_observation_numpy(self, visual_np, proprio_np):
        """Encode numpy observations to latent space (I added this now)"""
        # Convert numpy to tensor format expected by world model
        if visual_np.shape[2] == 3:  # HWC to CHW
            visual_np = np.transpose(visual_np, (2, 0, 1))
        
        # Normalize to world model format [0,1] -> [-1,1] (I added this now)
        if visual_np.max() > 1.0:
            visual_np = visual_np.astype(np.float32) / 255.0
        visual_np = 2.0 * visual_np - 1.0
        
        visual_tensor = torch.from_numpy(visual_np).unsqueeze(0).unsqueeze(0).to(self.device).float()
        proprio_tensor = torch.from_numpy(proprio_np).unsqueeze(0).unsqueeze(0).to(self.device).float()
        
        obs = {'visual': visual_tensor, 'proprio': proprio_tensor}
        return self.encode_observation_tensor(obs)
        
    def get_hj_value(self, visual_np, proprio_np):
        """Get best HJ value for current observation (I added this now)"""
        z = self.encode_observation_numpy(visual_np, proprio_np)
        with torch.no_grad():
            action = self.actor(z)
            hj_value = self.critic(z, action).item()
        return hj_value
    
    def get_hj_value_for_action(self, visual_np, proprio_np, action):
        """Get HJ value for specific action (I added this now)"""
        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action, dtype=torch.float32, device=self.device)
            if action.dim() == 0:
                action = action.unsqueeze(0)
        z = self.encode_observation_numpy(visual_np, proprio_np)
        with torch.no_grad():
            hj_value = self.critic(z, action).item()
        return hj_value

# -------------------------------------------------------------------------
# 4.  Occlusion helpers
# -------------------------------------------------------------------------

def generate_occlusion_mask(image_shape, patch_size, stride, i, j):
    y0, x0 = i * stride, j * stride
    y1, x1 = min(y0 + patch_size, image_shape[0]), min(x0 + patch_size, image_shape[1])
    mask = np.ones(image_shape, dtype=np.float32)
    mask[y0:y1, x0:x1, :] = 0
    return mask, (y0, y1, x0, x1)


def apply_occlusion(image, patch_size, stride, i, j,
                    method="black", blur_sigma=5):
    mask, (y0, y1, x0, x1) = generate_occlusion_mask(
        image.shape, patch_size, stride, i, j
    )

    if method == "black":
        return image * mask

    if method == "blur":
        out = image.copy()
        patch = to_uint8(out[y0:y1, x0:x1, :])
        blurred = cv2.GaussianBlur(patch, (0, 0), blur_sigma)
        out[y0:y1, x0:x1, :] = blurred.astype(np.float32) / 255.0
        return out

    raise ValueError(f"Unknown occlusion method {method!r}")

# -------------------------------------------------------------------------
# 5.  Visualisation helpers
# -------------------------------------------------------------------------

# def visualize_hj_saliency(image, saliency_map,
#                          baseline_hj, original_hj,
#                          save_path):
#     """Save original, heat‑map, and overlay panels for HJ saliency (I added this now)."""
#     fig, axes = plt.subplots(1, 3, figsize=(15, 5))

#     # -------- Original --------
#     img_vis = to_uint8(unnormalize(image).permute(1, 2, 0).cpu().numpy())
#     axes[0].imshow(img_vis, interpolation="bilinear", resample=True)
#     axes[0].set_title(
#         f"Original\nHJ Value: {baseline_hj:.3f}"  # I added this now
#     )
#     axes[0].axis("off")

#     # -------- Heat‑map --------
#     im1 = axes[1].imshow(
#         saliency_map, cmap="hot", interpolation="bilinear", resample=True
#     )
#     axes[1].set_title("HJ Saliency")  # I added this now
#     axes[1].axis("off")
#     plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

#     # -------- Overlay --------
#     sal_norm = (saliency_map - saliency_map.min()) / (
#         saliency_map.max() - saliency_map.min() + 1e-8
#     )
#     cmap = LinearSegmentedColormap.from_list(
#         "custom", [(0, 0, 1, 0.0), (1, 0, 0, 0.7)]
#     )
#     axes[2].imshow(img_vis, interpolation="bilinear", resample=True)
#     axes[2].imshow(
#         sal_norm, cmap=cmap, alpha=0.5, interpolation="bilinear", resample=True
#     )
#     axes[2].set_title("Overlay")
#     axes[2].axis("off")

#     plt.tight_layout()
#     plt.savefig(save_path, dpi=300, bbox_inches="tight")
#     plt.close()
from matplotlib.colors import TwoSlopeNorm

# def visualize_hj_saliency(image, saliency_map, baseline_hj, original_hj, save_path):
#     """Signed ΔHJ map: red = helpful region (occluding hurts HJ), blue = harmful region (occluding helps HJ)."""
#     fig, axes = plt.subplots(1, 3, figsize=(15, 5))

#     # -------- Original --------
#     img_vis = to_uint8(unnormalize(image).permute(1, 2, 0).cpu().numpy())
#     axes[0].imshow(img_vis, interpolation="bilinear", resample=True)
#     axes[0].set_title(f"Original\nHJ Value: {baseline_hj:.3f}")
#     axes[0].axis("off")

#     # -------- Heat-map (signed) --------
#     # symmetric scaling around 0
#     m = np.max(np.abs(saliency_map)) + 1e-8
#     norm = TwoSlopeNorm(vmin=-m, vcenter=0.0, vmax=m)
#     hm = axes[1].imshow(saliency_map, cmap="coolwarm", norm=norm, interpolation="bilinear", resample=True)
#     axes[1].set_title("ΔHJ (base − occluded)")
#     axes[1].axis("off")
#     cbar = plt.colorbar(hm, ax=axes[1], fraction=0.046, pad=0.04)
#     cbar.set_label("ΔHJ")

#     # -------- Overlay (alpha by magnitude, color by sign) --------
#     # alpha ∝ |ΔHJ|, clipped for readability
#     mag = np.abs(saliency_map) / (m + 1e-8)
#     mag = np.clip(mag, 0.0, 1.0)
#     axes[2].imshow(img_vis, interpolation="bilinear", resample=True)
#     axes[2].imshow(saliency_map, cmap="coolwarm", norm=norm, alpha=0.6*mag,
#                    interpolation="bilinear", resample=True)
#     axes[2].set_title("Overlay (color=sign, α=|ΔHJ|)")
#     axes[2].axis("off")

#     plt.tight_layout()
#     plt.savefig(save_path, dpi=300, bbox_inches="tight")
#     plt.close()
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import cv2
import matplotlib.pyplot as plt

def visualize_hj_saliency(
    image,
    saliency_map,
    baseline_hj,
    original_hj,
    save_path,
    *,
    use_grayscale_base=True,   # make base image grayscale so colors pop
    base_alpha=0.35,           # transparency of base image in overlay panel
    max_alpha=0.95,            # max overlay opacity
    min_alpha=0.15,            # min overlay opacity (nonzero so sign is still visible)
    alpha_gamma=0.7,           # <1 boosts mid-range values; >1 emphasizes peaks
    robust_pct=99.0,           # percentile for symmetric scaling around 0
    draw_contours=True,        # outlines for readability
    contour_levels=(0.5,),     # levels as fraction of vmax (|ΔHJ|)
    contour_color_pos="k",     # positive ΔHJ contour color
    contour_color_neg="w"      # negative ΔHJ contour color
):
    """Signed ΔHJ map: red = helpful (occluding lowers HJ), blue = harmful (occluding raises HJ)."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # -------- Original --------
    img_vis = to_uint8(unnormalize(image).permute(1, 2, 0).cpu().numpy())
    axes[0].imshow(img_vis, interpolation="bilinear", resample=True)
    axes[0].set_title(f"Original\nHJ Value: {baseline_hj:.3f}")
    axes[0].axis("off")

    # -------- Heat-map (signed, robust scaling) --------
    # symmetric scaling around 0 using robust max (percentile of |Δ|)
    abs_vals = np.abs(saliency_map).ravel()
    if abs_vals.size == 0 or np.all(abs_vals == 0):
        vmax = 1.0
    else:
        vmax = np.percentile(abs_vals, robust_pct)
        if vmax <= 0:
            vmax = np.max(abs_vals) + 1e-8
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    hm = axes[1].imshow(saliency_map, cmap="coolwarm", norm=norm,
                        interpolation="bilinear", resample=True)
    axes[1].set_title("ΔHJ (base − occluded)")
    axes[1].axis("off")
    cbar = plt.colorbar(hm, ax=axes[1], fraction=0.046, pad=0.04)
    cbar.set_label("ΔHJ")

    # -------- Overlay (fade base, color by sign, alpha by |Δ|^gamma) --------
    if use_grayscale_base:
        base_gray = cv2.cvtColor(img_vis, cv2.COLOR_RGB2GRAY)
        base_rgb = np.stack([base_gray, base_gray, base_gray], axis=-1)
    else:
        base_rgb = img_vis

    # show faded base
    axes[2].imshow(base_rgb, interpolation="bilinear", resample=True, alpha=base_alpha)

    # alpha ∝ |ΔHJ| with gamma + clipping
    mag = np.clip(np.abs(saliency_map) / (vmax + 1e-8), 0.0, 1.0)
    if alpha_gamma != 1.0:
        mag = mag ** alpha_gamma
    alpha_map = (min_alpha + (max_alpha - min_alpha) * mag).astype(np.float32)

    # draw colored saliency with per-pixel alpha
    axes[2].imshow(saliency_map, cmap="coolwarm", norm=norm,
                   interpolation="bilinear", resample=True, alpha=alpha_map)
    axes[2].set_title("Overlay (red=helpful, blue=harmful; α∝|ΔHJ|)")
    axes[2].axis("off")

    # optional contours around strong regions for readability
    if draw_contours and np.any(abs_vals > 0):
        # positive/negative masks at chosen contour levels (fractions of vmax)
        for frac in contour_levels:
            level = frac * vmax
            if level > 0:
                # positive
                try:
                    cs_pos = axes[2].contour(saliency_map, levels=[level], colors=contour_color_pos, linewidths=1.0)
                    # negative
                    cs_neg = axes[2].contour(saliency_map, levels=[-level], colors=contour_color_neg, linewidths=1.0)
                except Exception:
                    pass

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def create_occlusion_grid(image, patch_size=16, stride=8,
                          output_path=None, method="black", blur_sigma=5):
    """3×3 montage showing exemplar occlusions."""
    img_np = unnormalize(image).permute(1, 2, 0).cpu().numpy()
    H, W = img_np.shape[:2]
    n_rows = (H - patch_size) // stride + 1
    n_cols = (W - patch_size) // stride + 1

    positions = [
        (0, 0, "Top‑left"),      (0, n_cols // 2, "Top‑centre"),    (0, n_cols - 1, "Top‑right"),
        (n_rows // 2, 0, "Mid‑left"), (n_rows // 2, n_cols // 2, "Centre"), (n_rows // 2, n_cols - 1, "Mid‑right"),
        (n_rows - 1, 0, "Bottom‑left"), (n_rows - 1, n_cols // 2, "Bottom‑centre"), (n_rows - 1, n_cols - 1, "Bottom‑right")
    ]

    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    axes = axes.ravel()

    for k, (i, j, label) in enumerate(positions):
        occ = apply_occlusion(
            img_np, patch_size, stride, i, j,
            method=method, blur_sigma=blur_sigma
        )
        axes[k].imshow(to_uint8(occ), interpolation="bilinear", resample=True)
        axes[k].add_patch(
            plt.Rectangle((j * stride, i * stride), patch_size, patch_size,
                          linewidth=2, edgecolor="red", facecolor="none")
        )
        axes[k].set_title(label, fontsize=10)
        axes[k].axis("off")

    plt.suptitle(f"Occlusion examples ({method}, {patch_size}px)", fontsize=14)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

# -------------------------------------------------------------------------
# 6.  HJ-based saliency computation (I added this now)
# -------------------------------------------------------------------------

def compute_saliency_map(hj_evaluator, image, proprio,
                         patch_size=16, stride=8,
                         device="cuda",
                         save_examples=False, sample_idx=0,
                         out_dir=None,
                         method="black", blur_sigma=5):
    """Returns (saliency_map [H×W], baseline_hj_value) (I added this now)."""
    hj_evaluator.actor.eval()
    hj_evaluator.critic.eval()

    # Get baseline HJ value (I added this now)
    img_np = unnormalize(image).permute(1, 2, 0).cpu().numpy()
    proprio_np = proprio.cpu().numpy()
    base_hj = hj_evaluator.get_hj_value(img_np, proprio_np)

    H, W = image.shape[1:]
    n_rows = (H - patch_size) // stride + 1
    n_cols = (W - patch_size) // stride + 1
    scores = np.zeros((n_rows, n_cols), dtype=np.float32)

    sample_pos = []
    if save_examples:
        sample_pos = [
            (0, 0, "tl"), (0, n_cols - 1, "tr"),
            (n_rows // 2, n_cols // 2, "ctr"),
            (n_rows - 1, 0, "bl"), (n_rows - 1, n_cols - 1, "br")
        ]

    for i in tqdm(range(n_rows), desc="Computing HJ saliency"):
        for j in range(n_cols):
            occ_img = apply_occlusion(
                img_np, patch_size, stride, i, j,
                method=method, blur_sigma=blur_sigma
            )
            
            # Get HJ value for occluded image (I added this now)
            occ_hj = hj_evaluator.get_hj_value(occ_img, proprio_np)
            
            # Saliency is the change in HJ value (I added this now)
            scores[i, j] = base_hj - occ_hj

            # optional snapshots
            if save_examples and out_dir and any(p[:2] == (i, j) for p in sample_pos):
                tag = [p[2] for p in sample_pos if p[0] == i and p[1] == j][0]
                plt.figure(figsize=(4, 4))
                plt.imshow(to_uint8(occ_img), interpolation="bilinear", resample=True)
                plt.axis("off")
                plt.title(f"{tag} HJ_drop={scores[i, j]:.3f}")  # I added this now
                plt.savefig(Path(out_dir) / f"occ_{sample_idx:03d}_{tag}_{method}.png",
                            dpi=150, bbox_inches="tight")
                plt.close()

    saliency = cv2.resize(scores, (W, H), interpolation=cv2.INTER_LINEAR)
    return saliency, base_hj  # I added this now - return HJ value instead of class

# -------------------------------------------------------------------------
# 7.  Data loading
# -------------------------------------------------------------------------

def load_data(task: str, root: str, seed=42, n_rollout=None, split_ratio=0.9):
    """
    Returns (train_slices, val_slices, train_slices_cost, val_slices_cost)
    where slices are `TrajSlicerDataset` objects created by
    `get_train_val_sliced_with_cost`.
    """
    if "dubins" in task:
        from datasets.dubins_dset import PointMazeDataset as Dataset
    elif "maniskill" in task:
        from datasets.maniskill_dset import ManiSkillDataset as Dataset
    elif "carla" in task:
        from datasets.carla_dset import CarlaDataset as Dataset
    elif "cargoal" in task:
        from datasets.cargoal_dset import PointMazeDataset as Dataset
    else:
        raise ValueError("dataset not supported")

    dset_path = Path(root) / "dubins1800_continuous_cost"
    base_dset = Dataset(
        str(dset_path),
        transform=default_transform(),
        normalize_action=True,
        normalize_states=True,
        n_rollout=n_rollout,
        with_costs=True
    )
    cost_dset = Dataset(
        str(dset_path),
        transform=default_transform(),
        normalize_action=True,
        normalize_states=True,
        n_rollout=n_rollout,
        with_costs=True,
        only_cost=True
    )

    train_s, val_s, train_c, val_c = get_train_val_sliced_with_cost(
        traj_dataset=base_dset,
        traj_dataset_cost=cost_dset,
        train_fraction=split_ratio,
        num_frames=1,
        random_seed=seed,
        frameskip=1
    )
    return train_s, val_s, train_c, val_c

# -------------------------------------------------------------------------
# 8.  Main
# -------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser("HJ Occlusion‑based saliency")  # I added this now
    p.add_argument("--task",         default="maniskill3000classif")
    p.add_argument("--data_path",    required=True)
    p.add_argument("--backbone",     default="r3m")
    p.add_argument("--device",       default="cuda")
    p.add_argument("--with_proprio", action="store_true")
    p.add_argument("--num_samples",  type=int, default=10)
    p.add_argument("--patch_size",   type=int, default=16)
    p.add_argument("--stride",       type=int, default=8)
    p.add_argument("--seed",         type=int, default=1)
    p.add_argument("--occlusion_method", choices=["black", "blur"], default="black")
    p.add_argument("--output_dir",   required=True)
    p.add_argument("--blur_sigma",   type=float, default=10.0)
    p.add_argument("--wm_ckpt_root", default="checkpoints/wm/dubins",
                   help="Root directory of the DINO-WM checkpoints (final dir is <root>/<backbone>)")
    p.add_argument("--hj_ckpt_dir",  default="checkpoints/hj/dubins",
                   help="Root directory of the trained HJ checkpoints")
    args = p.parse_args()

    # I added this now to log to a good directory to separate results
    args.output_dir = f"{args.output_dir}/{args.backbone}"
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    # ---------- world‑model ----------
    ckpt_dir = Path(args.wm_ckpt_root) / args.backbone
    
    hydra_cfg = ckpt_dir / 'hydra.yaml'
    wm_snapshot = ckpt_dir / 'checkpoints' / 'model_latest.pth'
    
    train_cfg = OmegaConf.load(str(hydra_cfg))
    num_action_repeat = train_cfg.num_action_repeat
    
    wm = load_model(wm_snapshot, train_cfg, num_action_repeat, device=args.device)
    wm.eval()
    
    # ---------- HJ Policy Evaluator (I added this now) ----------
    hj_ckpt_dir = Path(args.hj_ckpt_dir) / args.backbone / "epoch_200"
    actor_path = hj_ckpt_dir / "actor.pth"
    critic_path = hj_ckpt_dir / "critic.pth"
    
    # Create HJ evaluator (I added this now)
    hj_evaluator = HJPolicyEvaluator(str(actor_path), str(critic_path), wm, args.device, with_proprio=args.with_proprio)
    print(f"Loaded HJ policy from {hj_ckpt_dir}")

    # ---------- gather samples ----------
    print(f"[INFO] Generating HJ saliency maps for {args.num_samples} frames …")
    
    # Load data
    _, val_slices, _, _ = load_data(args.task, args.data_path, args.seed)
    val_loader = torch.utils.data.DataLoader(val_slices, batch_size=1, shuffle=True)
    
    samples = []

    for obs, _, _, cost, _ in val_loader: #found in file TrajSlicerWithCostDataset in sentence return tuple([obs, act, state, cost, info["h"][start:end:self.frameskip]])
        obs_gpu = {k: v.to(args.device) for k, v in obs.items()}
        
        # Get HJ value for this observation (I added this now)
        img_np = unnormalize(obs_gpu["visual"].squeeze(0).squeeze(0)).permute(1, 2, 0).cpu().numpy()
        proprio_np = obs_gpu["proprio"].squeeze(0).squeeze(0).cpu().numpy()
        hj_value = hj_evaluator.get_hj_value(img_np, proprio_np)
        
        true_lbl = cost.item()
        
        # Collect samples (I added this now)
        samples.append((obs, true_lbl, hj_value))
        
        if len(samples) >= args.num_samples: 
            break

    print(f"Collected {len(samples)} samples")

    # Process samples (I added this now)
    for idx, (obs, true_lbl, hj_value) in enumerate(samples):
        img  = obs["visual"].squeeze(0).squeeze(0)
        prop = obs["proprio"].squeeze(0).squeeze(0)

        sal, baseline_hj = compute_saliency_map(  # I added this now
            hj_evaluator, img, prop,
            patch_size=args.patch_size, stride=args.stride,
            device=args.device,
            save_examples=(idx < 5), sample_idx=idx,
            out_dir=args.output_dir,
            method=args.occlusion_method, blur_sigma=args.blur_sigma
        )

        out_png = Path(args.output_dir) / (
            f"hj_sal_{idx:03d}_baseline{baseline_hj:.3f}_"  # I added this now
            f"true{'unsafe' if true_lbl else 'safe'}.png"
        )
        
        # Update visualization call (I added this now)
        visualize_hj_saliency(img, sal, baseline_hj, hj_value, out_png)

        if idx < 5:
            grid_png = Path(args.output_dir) / (
                f"grid_{idx:03d}_{args.occlusion_method}.png"
            )
            create_occlusion_grid(
                img, patch_size=args.patch_size, stride=args.stride,
                output_path=grid_png,
                method=args.occlusion_method, blur_sigma=args.blur_sigma
            )
        print(f"[{idx+1}/{len(samples)}] saved {out_png.name}, HJ: {baseline_hj:.3f}")

    print(f"\n✅  All HJ saliency outputs saved under: {args.output_dir}")

# -------------------------------------------------------------------------
if __name__ == "__main__":
    main()
    

# ΔHJ > 0 (plotted as red/orange in your current “hot” colormap)
# means the occluded HJ value is lower than the baseline.
# In words: when you remove (occlude) this patch, the system’s HJ value drops.
# → That patch was supporting a higher / safer value.

# ΔHJ < 0 (dark/blue, if you use a diverging colormap)
# means the occluded HJ value is higher than the baseline.
# In words: removing this patch makes things look safer to the critic.
# → That patch was hurting the value.

# ΔHJ ≈ 0
# means the patch is irrelevant for the value estimate.


    
    #python saliency_new_dubins.py --task dubins --data_path "datasets" --backbone r3m --occlusion_method blur --blur_sigma 11 --num_samples 20 --output_dir outputs/saliency/dubins --seed 3 --patch_size 20 --stride 10 --with_proprio
    #python saliency_new_dubins.py --task dubins --data_path "datasets" --backbone r3m --occlusion_method black --blur_sigma 11 --num_samples 20 --output_dir outputs/saliency/dubins --seed 3 --patch_size 20 --stride 10 --with_proprio
    