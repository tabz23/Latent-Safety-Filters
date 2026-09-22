
import torch
import decord
import numpy as np
from pathlib import Path
from einops import rearrange
from decord import VideoReader
from typing import Callable, Optional
from .traj_dset import TrajDataset, get_train_val_sliced
decord.bridge.set_bridge("torch")

class PointMazeDataset(TrajDataset):
    def __init__(
        self,
        data_path: str = "",
        n_rollout: Optional[int] = None,
        transform: Optional[Callable] = None,
        normalize_action: bool = False,
        normalize_states: bool = True,
        with_costs: bool = True,
        action_scale=1.0,
        only_cost: bool = False,
    ):
        self.data_path = Path(data_path)
        self.transform = transform
        self.normalize_action = normalize_action
        self.only_cost = only_cost

        # Load dataset from disk (list of tensors)
        self.states = torch.load(self.data_path / "states.pth")  # list of [T_i, 9]
        self.actions = torch.load(self.data_path / "actions.pth")  # list of [T_i, 1]
        self.seq_lengths = torch.load(self.data_path / 'seq_lengths.pth')  # list of ints
        self.costs = torch.load(self.data_path / "costs.pth")

        # Truncate rollouts if needed
        if n_rollout:
            self.states = self.states[:n_rollout]
            self.actions = self.actions[:n_rollout]
            self.seq_lengths = self.seq_lengths[:n_rollout]

        # Cast to float and scale
        self.states = [s.float() for s in self.states]
        self.actions = [a.float() / action_scale for a in self.actions]
        self.proprios = [s[:, :24].clone() for s in self.states]

        print(f"Loaded {len(self.states)} rollouts")

        self.n_rollout = len(self.states)
        self.action_dim = self.actions[0].shape[-1]
        self.state_dim = self.states[0].shape[-1]
        self.proprio_dim = self.proprios[0].shape[-1]

        if normalize_action:
            self.action_mean, self.action_std = self.get_data_mean_std(self.actions)
            self.state_mean, self.state_std = self.get_data_mean_std(self.states)
            self.proprio_mean, self.proprio_std = self.get_data_mean_std(self.proprios)
        else:
            self.action_mean = torch.zeros(self.action_dim)
            self.action_std = torch.ones(self.action_dim)
            self.state_mean = torch.zeros(self.state_dim)
            self.state_std = torch.ones(self.state_dim)
            self.proprio_mean = torch.zeros(self.proprio_dim)
            self.proprio_std = torch.ones(self.proprio_dim)

        self.actions = [(a - self.action_mean) / self.action_std for a in self.actions]
        self.proprios = [(p - self.proprio_mean) / self.proprio_std for p in self.proprios]

    def get_data_mean_std(self, data_list):
        all_data = torch.cat(data_list, dim=0)
        return torch.mean(all_data, dim=0), torch.std(all_data, dim=0)

    def get_seq_length(self, idx):
        return self.seq_lengths[idx]

    def get_all_actions(self):
        return torch.cat([a[:T] for a, T in zip(self.actions, self.seq_lengths)], dim=0)

    # def get_frames(self, idx, frames):
    #     obs_dir = self.data_path / "obses"
    #     image = torch.load(obs_dir / f"episode_{idx:03d}.pth")  # [T, 224, 224, 3]

    #     # Index sequence
    #     image = image[frames]  # THWC
    #     image = image / 255.0
    #     image = rearrange(image, "T H W C -> T C H W")
    #     if self.transform:
    #         image = self.transform(image)

    #     proprio = self.proprios[idx][frames]
    #     act = self.actions[idx][frames]
    #     state = self.states[idx][frames]

    #     obs = {
    #         "visual": image,
    #         "proprio": proprio
    #     }
    #     return obs, act, state, {}  # env_info placeholder
    def get_frames(self, idx, frames):
        if not self.only_cost:
            obs_dir = self.data_path / "obses"
            image = torch.load(obs_dir / f"episode_{idx:03d}.pth")  # [T, 224, 224, 3]
            
            # Convert frames to list for easier debugging
            if isinstance(frames, range):
                frame_list = list(frames)
            else:
                frame_list = frames
            
            # Debug information
            seq_len = self.get_seq_length(idx)
            image_len = image.shape[0]
            
            try:
                # Index sequence
                image = image[frames]  # THWC
            except IndexError as e:
                print(f"ERROR in episode {idx}:")
                print(f"  Image shape: {image.shape}")
                print(f"  Stored sequence length: {seq_len}")
                print(f"  Actual image frames: {image_len}")
                print(f"  Frames requested: {frames}")
                if isinstance(frames, range):
                    print(f"  Frames range: min={min(frame_list)}, max={max(frame_list)}, count={len(frame_list)}")
                print(f"  Original error: {e}")
                raise e  # Re-raise the error after logging

            image = image / 255.0
            image = rearrange(image, "T H W C -> T C H W")

            if self.transform:
                image = self.transform(image)

            proprio = self.proprios[idx][frames]
            act = self.actions[idx][frames]
            state = self.states[idx][frames]

            obs = {
                "visual": image,
                "proprio": proprio
            }
            # print(f"obs['visual'] shape: {obs['visual'].shape}")
            # print(f"obs['proprio'] shape: {obs['proprio'].shape}")
            # print(f"actions shape: {act.shape}")
            # print(f"full_states shape: {state.shape}")
            # # print(f"self.costs shape: {costs.shape}")
            # print(f"self.costs[idx] shape: {self.costs[idx].shape if hasattr(self.costs[idx], 'shape') else 'scalar'}")
            # print(f"(self.costs[idx]>0).long() shape: {(self.costs[idx]>0).long().shape if hasattr((self.costs[idx]>0).long(), 'shape') else 'scalar'}")
            # print(f"frames: {frames}")
            # print(f"idx: {idx}")
            # print("---")
            return obs, act, state, {"cost":(self.costs[idx]<=0).long(),"h":self.costs[idx]}  # env_info placeholder
        else:
            return None, None, None, {"cost":(self.costs[idx]<=0).long(),"h":self.costs[idx]}

    def __getitem__(self, idx):
        return self.get_frames(idx, range(self.get_seq_length(idx)))

    def __len__(self):
        return len(self.seq_lengths)

    def preprocess_imgs(self, imgs):
        if isinstance(imgs, np.ndarray):
            raise NotImplementedError
        elif isinstance(imgs, torch.Tensor):
            return rearrange(imgs, "b h w c -> b c h w") / 255.0

def load_cargoal_slice_train_val(
    transform,
    n_rollout=50,
    data_path='',
    normalize_action=False,
    split_ratio=0.8,
    num_hist=0,
    num_pred=0,
    frameskip=0,
):
    dset = PointMazeDataset(
        n_rollout=n_rollout,
        transform=transform,
        data_path=data_path,
        normalize_action=normalize_action,
    )
    dset_train, dset_val, train_slices, val_slices = get_train_val_sliced(
        traj_dataset=dset, 
        train_fraction=split_ratio, 
        num_frames=num_hist + num_pred, 
        frameskip=frameskip
    )

    datasets = {
        'train': train_slices,
        'valid': val_slices
    }
    traj_dset = {
        'train': dset_train,
        'valid': dset_val
    }
    return datasets, traj_dset
