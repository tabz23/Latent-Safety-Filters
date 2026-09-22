# import torch
# import numpy as np
# from pathlib import Path
# from einops import rearrange
# from typing import Callable, Optional
# from datasets.traj_dset import TrajDataset, TrajSlicerDataset, get_train_val_sliced, split_traj_datasets



# class ManiSkillDataset(TrajDataset):
#     def __init__(
#         self,
#         data_path: str,
#         transform: Optional[Callable] = None,
#         normalize_action: bool = True,
#         normalize_states: bool = True,
#         n_rollout: Optional[int] = None,
#         with_costs: bool = True
#     ):
#         self.data_path = Path(data_path)
#         self.transform = transform
#         self.normalize_action = normalize_action
#         self.normalize_states = normalize_states
#         self.with_costs = with_costs

#         self.states = torch.load(self.data_path / "states.pth")  # list of [T_i, 9]
#         self.actions = torch.load(self.data_path / "actions.pth")  # list of [T_i, 1]
#         self.seq_lengths = torch.load(self.data_path / 'seq_lengths.pth')  # list of ints

#         self.n_rollout = n_rollout
#         if self.n_rollout:
#             n = min(n_rollout, len(self.seq_lengths))
#         else:
#             n = len(self.states)

#         self.states = self.states[:n]
#         self.actions = self.actions[:n]
#         self.seq_lengths = self.seq_lengths[:n]
        
#         self.proprios = self.states[..., 26:58].clone()

#         self.action_dim = self.actions.shape[-1]
#         self.state_dim = self.states.shape[-1]
#         self.proprio_dim = self.proprios.shape[-1]
        
#         if normalize_action:
#             self.action_mean, self.action_std = self.get_data_mean_std(self.actions, self.seq_lengths)
#             self.state_mean, self.state_std = self.get_data_mean_std(self.states, self.seq_lengths)
#             self.proprio_mean, self.proprio_std = self.get_data_mean_std(self.proprios, self.seq_lengths)
#         else:
#             self.action_mean = torch.zeros(self.action_dim)
#             self.action_std = torch.ones(self.action_dim)
#             self.state_mean = torch.zeros(self.state_dim)
#             self.state_std = torch.ones(self.state_dim)
#             self.proprio_mean = torch.zeros(self.proprio_dim)
#             self.proprio_std = torch.ones(self.proprio_dim)

#         self.actions = (self.actions - self.action_mean) / self.action_std
#         self.proprios = (self.proprios - self.proprio_mean) / self.proprio_std



#     def get_seq_length(self, idx):
#         return self.seq_lengths[idx].item()

#     def get_all_actions(self):
#         result = []
#         for i in range(len(self.seq_lengths)):
#             T = self.get_seq_length(i)
#             result.append(self.actions[i, :T, :])
#         return torch.cat(result, dim = 0)

#     def get_frames(self, idx, frames):
#         obs_file = self.data_path / "obses" / f"episode_{idx}.pth"
#         images = torch.load(obs_file, map_location="cpu")
#         images = images.float() / 255.0
#         images = rearrange(images, "T H W C -> T C H W")
#         if self.transform:
#             images = self.transform(images)

#         image = images[frames]
#         actions = self.actions[idx, frames]
#         full_states = self.states[idx, frames]
#         proprio_states = self.proprios[idx, frames]
            
#         obs = {
#             "visual" : image,
#             "proprio" : proprio_states
#         }

#         return obs, actions, full_states, {}
        

#     def __getitem__(self, idx):
#         return self.get_frames(idx, range(self.get_seq_length(idx)))
    
#     def __len__(self):
#         return len(self.seq_lengths)
    
#     def preprocess_imgs(self, imgs):
#         if isinstance(imgs, np.ndarray):
#             raise NotImplementedError
#         elif isinstance(imgs, torch.Tensor):
#             return rearrange(imgs, "b h w c -> b c h w") / 255.0
        
#     def get_data_mean_std(self, data, traj_lengths):
#         all_data = []
#         for traj in range(len(traj_lengths)):
#             traj_len = traj_lengths[traj]
#             traj_data = data[traj, :traj_len]
#             all_data.append(traj_data)
#         all_data = torch.vstack(all_data)
#         data_mean = torch.mean(all_data, dim=0)
#         data_std = torch.std(all_data, dim=0)
#         return data_mean, data_std
        
# def load_maniskill_slice_train_val(
#         transform,
#         data_path,
#         n_rollout = 50,
#         normalize_action = True,
#         normalize_states = True,
#         split_ratio = 0.8,
#         num_hist = 0,
#         num_pred = 0,
#         frameskip = 0,
#         with_costs = True,
# ):
#     dset = ManiSkillDataset(
#         data_path=data_path,
#         transform=transform,
#         normalize_action=normalize_action,
#         normalize_states=normalize_states,
#         n_rollout=n_rollout,
#         with_costs = with_costs
#     )

#     train_dset, val_dset = split_traj_datasets(
#         dset,
#         train_fraction=split_ratio
#     )
    
#     dset_train, dset_val, train_slices, val_slices = get_train_val_sliced(
#         traj_dataset=dset, 
#         train_fraction=split_ratio, 
#         num_frames=num_hist + num_pred, 
#         frameskip=frameskip
#     )

#     datasets = {}
#     datasets['train'] = train_slices
#     datasets['valid'] = val_slices
#     traj_dset = {}
#     traj_dset['train'] = dset_train
#     traj_dset['valid'] = dset_val
#     return datasets, traj_dset

import torch
import numpy as np
from pathlib import Path
from einops import rearrange
from typing import Callable, Optional
from datasets.traj_dset import TrajDataset, TrajSlicerDataset, get_train_val_sliced, split_traj_datasets



class ManiSkillDataset(TrajDataset):
    def __init__(
        self,
        data_path: str,
        transform: Optional[Callable] = None,
        normalize_action: bool = True,
        normalize_states: bool = True,
        n_rollout: Optional[int] = None,
        with_costs: bool = True,
        only_cost: bool = False,
    ):
        self.data_path = Path(data_path)
        self.transform = transform
        self.normalize_action = normalize_action
        self.normalize_states = normalize_states
        self.with_costs = with_costs
        self.only_cost = only_cost

        self.states = torch.load(self.data_path / "states.pth").float()
        self.actions = torch.load(self.data_path / "actions.pth").float()
        self.seq_lengths = torch.load(self.data_path / "seq_lengths.pth").long()
        self.costs = torch.load(self.data_path / "costs.pth").float()

        self.n_rollout = n_rollout
        if self.n_rollout:
            n = min(n_rollout, len(self.seq_lengths))
        else:
            n = len(self.states)

        self.states = self.states[:n]
        self.actions = self.actions[:n]
        self.seq_lengths = self.seq_lengths[:n]
        
        self.proprios = self.states[..., 39:89].clone()

        self.action_dim = self.actions.shape[-1]
        self.state_dim = self.states.shape[-1]
        self.proprio_dim = self.proprios.shape[-1]
        
        if normalize_action:
            self.action_mean, self.action_std = self.get_data_mean_std(self.actions, self.seq_lengths)
            self.state_mean, self.state_std = self.get_data_mean_std(self.states, self.seq_lengths)
            self.proprio_mean, self.proprio_std = self.get_data_mean_std(self.proprios, self.seq_lengths)
        else:
            self.action_mean = torch.zeros(self.action_dim)
            self.action_std = torch.ones(self.action_dim)
            self.state_mean = torch.zeros(self.state_dim)
            self.state_std = torch.ones(self.state_dim)
            self.proprio_mean = torch.zeros(self.proprio_dim)
            self.proprio_std = torch.ones(self.proprio_dim)

        self.actions = (self.actions - self.action_mean) / self.action_std
        self.proprios = (self.proprios - self.proprio_mean) / self.proprio_std



    def get_seq_length(self, idx):
        return self.seq_lengths[idx].item()

    def get_all_actions(self):
        result = []
        for i in range(len(self.seq_lengths)):
            T = self.get_seq_length(i)
            result.append(self.actions[i, :T, :])
        return torch.cat(result, dim = 0)

    def get_frames(self, idx, frames):
        if not self.only_cost:
            obs_file = self.data_path / "obses" / f"episode_{idx}.pth"
            images = torch.load(obs_file, map_location="cpu")
            images = images.float() / 255.0
            images = rearrange(images, "T H W C -> T C H W")
            if self.transform:
                images = self.transform(images)

            image = images[frames]
            actions = self.actions[idx, frames]
            full_states = self.states[idx, frames]
            proprio_states = self.proprios[idx, frames]
                
            obs = {
                "visual" : image,
                "proprio" : proprio_states
            }

            return obs, actions, full_states, {"cost":(self.costs[idx]>0).long(),"h":-self.costs[idx]+0.5}
        else:
            return None, None, None, {"cost":(self.costs[idx]>0).long(),"h":-self.costs[idx]+0.5}
        

    def __getitem__(self, idx):
        return self.get_frames(idx, range(self.get_seq_length(idx)))
    
    def __len__(self):
        return len(self.seq_lengths)
    
    def preprocess_imgs(self, imgs):
        if isinstance(imgs, np.ndarray):
            raise NotImplementedError
        elif isinstance(imgs, torch.Tensor):
            return rearrange(imgs, "b h w c -> b c h w") / 255.0
        
    def get_data_mean_std(self, data, traj_lengths):
        all_data = []
        for traj in range(len(traj_lengths)):
            traj_len = traj_lengths[traj]
            traj_data = data[traj, :traj_len]
            all_data.append(traj_data)
        all_data = torch.vstack(all_data)
        data_mean = torch.mean(all_data, dim=0)
        data_std = torch.std(all_data, dim=0)
        return data_mean, data_std
        
def load_maniskill_slice_train_val(
        transform,
        data_path,
        n_rollout = 50,
        normalize_action = True,
        normalize_states = True,
        split_ratio = 0.8,
        num_hist = 0,
        num_pred = 0,
        frameskip = 0,
        with_costs = True,
):
    dset = ManiSkillDataset(
        data_path=data_path,
        transform=transform,
        normalize_action=normalize_action,
        normalize_states=normalize_states,
        n_rollout=n_rollout,
        with_costs = with_costs
    )

    train_dset, val_dset = split_traj_datasets(
        dset,
        train_fraction=split_ratio
    )
    
    dset_train, dset_val, train_slices, val_slices = get_train_val_sliced(
        traj_dataset=dset, 
        train_fraction=split_ratio, 
        num_frames=num_hist + num_pred, 
        frameskip=frameskip
    )

    datasets = {}
    datasets['train'] = train_slices
    datasets['valid'] = val_slices
    traj_dset = {}
    traj_dset['train'] = dset_train
    traj_dset['valid'] = dset_val
    return datasets, traj_dset
