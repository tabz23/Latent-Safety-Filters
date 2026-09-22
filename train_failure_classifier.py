from plan import load_model
from omegaconf import OmegaConf
import argparse
import yaml
from pathlib import Path
import torch
from datasets.traj_dset import split_traj_datasets, get_train_val_sliced_with_cost
from datasets.img_transforms import default_transform
from tqdm import tqdm, trange
import wandb
import os
import numpy as np
from sklearn.metrics import confusion_matrix
from copy import deepcopy
from scipy.stats import pearsonr

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
    # 1) Top‐level parser for the flags you always need
    parser = argparse.ArgumentParser("Failure Classifier Training")
    parser.add_argument(
        "--dino_ckpt_dir", type=str,
        default="checkpoints/wm",
        help="Where to find the DINO-WM checkpoints"
    )
    parser.add_argument(
        "--data_root", type=str,
        default="datasets",
        help="Root directory of the datasets (each task's dataset lives in a subdirectory)"
    )
    parser.add_argument(
        "--output_root", type=str,
        default="checkpoints/classifier",
        help="Where to save the trained failure classifier checkpoints"
    )
    parser.add_argument(
        "--config", type=str, default="train_failure_classifier.yaml",
        help="Path to your flat YAML of hyperparameters"
    )
    
    parser.add_argument(
    "--without_proprio", default=False, action="store_true",
    help="Flag to include proprioceptive information in latent encoding"
    )
    
    parser.add_argument(
    "--dino_encoder", type=str, default="dino",
    help="Which encoder to use: dino, r3m, vc1, etc."
    )
    
    parser.add_argument(
    "--task", type=str, default="maniskillnew",
    help="Which task to perform: dubins, other_task, etc."
    )

    parser.add_argument(
    "--finetune", default=False, action='store_true',
    )

    parser.add_argument(
    "--single_layer_classifier", default=False, action='store_true',
    )
    parser.add_argument(
    "--epochs", type=int, default=2,
    help="Number of training epochs"
    )
    parser.add_argument(
    "--backbones", type=str, nargs="+", default=None,
    help="Which backbones to train (default: all paper backbones)"
    )

    

    args, remaining = parser.parse_known_args()

    # 2) Load all keys & values from the YAML (no `defaults:` wrapper needed)
    with open(args.config) as f:
        cfg = yaml.safe_load(f)  # e.g. {'batch_size':16, 'critic-lr':1e-3, ...}

    # 3) Dynamically build a second parser so each key adopts the right type
    cfg_parser = argparse.ArgumentParser()
    for key, val in sorted(cfg.items()):
        arg_t = args_type(val)
        cfg_parser.add_argument(f"--{key}", type=arg_t, default=arg_t(val))
    cfg_args = cfg_parser.parse_args(remaining)

    # 4) Merge everything back into the top‐level args namespace
    for key, val in vars(cfg_args).items():
        if not hasattr(args, key) or getattr(args, key) == parser.get_default(key):
            setattr(args, key.replace("-", "_"), val)

    return args

class FailureClassifier(torch.nn.Module):
    def __init__(self, sample, wm, args):
        super(FailureClassifier, self).__init__()
        self.wm = wm
        self.args = args
        if self.args.freeze_wm:
            for p in self.wm.parameters():
                p.requires_grad = False
        ''' As hussein said, we need a single layer classifier to evaluate the performance of the visual representation. Here I didn't use sigmoid type output because I notice you concated the representations of all tokens together to construct the final representation. If we use weighted sum to calculate the final value. The trained weight will be very small, which is not good intuitively.
        
        For multi-layer one, I use the same setting used in the latent safety filter paper
        '''
        hidden_dim = self.encode(sample).shape[-1]
        print(hidden_dim)
        if self.args.single_layer_classifier:
            self.head = torch.nn.Linear(hidden_dim, 2)
        else:
            self.head = torch.nn.Sequential(
                torch.nn.Linear(hidden_dim, 512),
                torch.nn.ReLU(),
                torch.nn.Linear(512, 256),
                torch.nn.ReLU(),
                torch.nn.Linear(256, 1)
            )

    def forward(self, obs):
        rep = self.encode(obs)
        return self.head(rep)

    def encode(self, obs):
        vis = obs['visual'].unsqueeze(1)
        proprio = obs['proprio'].unsqueeze(1)
        B = vis.shape[0]
        data = {'visual': vis, 'proprio': proprio}
        latent = self.wm.encode_obs(data)
        if self.args.with_proprio:
            lat_vis = latent['visual'].reshape(B, -1)
            lat_prop = latent['proprio'].reshape(B, -1)
            rep = torch.cat([lat_vis, lat_prop], dim=-1)
        else:
            rep = latent['visual'].reshape(B, -1)
        return rep

class FailureClassifierDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, mode = "train"):
        path = data_path / f"{mode}"
        self.data_path = Path(path)
        # need to store data in three dirs: train, val, test
        # then three pth files store visual, proprio, labels
        # How do we collect the safe trajectories in each task?
        self.visual = torch.load(self.data_path / "visual.pth") 
        self.proprio = torch.load(self.data_path / "proprio.pth")
        self.labels = torch.load(self.data_path / "labels.pth")

    def __len__(self):
        return len(self.visual)

    def __getitem__(self, idx):
        visual = self.visual[idx]
        proprio = self.proprio[idx]
        visual = torch.tensor(visual, dtype=torch.float32)
        # DEBUG: print your min/max once
        # print(f"[DEBUG] visual raw min={visual.min():.0f}, max={visual.max():.0f}")
        proprio = torch.tensor(proprio, dtype=torch.float32)
        obs = {'visual': visual/255, 'proprio': proprio}
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return {'obs': obs, 'labels': label}

def load_data(
        task,
        data_path,
        seed = 42,
        transform = default_transform,
        n_rollout = None,
        normalize_action = False,
        normalize_states = True,
        split_ratio = 0.9,
        num_hist = 1,
        num_pred = 0,
        frameskip = 1,
        with_costs = True,
        ):
    if "dubins" in task:
        from datasets.dubins_dset import PointMazeDataset as Dataset
    elif "maniskill" in task:
        from datasets.maniskill_dset import ManiSkillDataset as Dataset
    elif "carla" in task:
        from datasets.carla_dset import CarlaDataset as Dataset
    elif "cargoal" in task:
        from datasets.cargoal_dset import PointMazeDataset as Dataset
    else:
        raise("dataset not supported")
    path = Path(data_path) / task
    dset = Dataset(
        data_path=path,
        transform=default_transform(),
        normalize_action=normalize_action,
        normalize_states=normalize_states,
        n_rollout=n_rollout,
        with_costs=with_costs
    )
    dset_cost = Dataset(
        data_path=path,
        transform=default_transform(),
        normalize_action=normalize_action,
        normalize_states=normalize_states,
        n_rollout=n_rollout,
        with_costs=with_costs,
        only_cost=True,
    )
    train_slices, val_slices, train_slices_cost, val_slices_cost = get_train_val_sliced_with_cost(
        traj_dataset=dset, 
        traj_dataset_cost=dset_cost, 
        train_fraction=split_ratio, 
        num_frames=num_hist + num_pred, 
        random_seed=seed,
        frameskip=frameskip
    )
    return train_slices, val_slices, train_slices_cost, val_slices_cost

def train(fc, train_dataloader, val_dataloader, args, logger):
    optimizer = torch.optim.Adam(fc.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    train_stats = {'loss':[],'loss_safe':[],'loss_unsafe':[],'loss_unsafe_weak':[],'loss_gradient_penalty':[]}
    val_stats = {}
    fc.train()
    for _ in trange(args.epochs):
        pbar = tqdm(train_dataloader)
        for idx, batch in enumerate(pbar):
            optimizer.zero_grad()
            obs, act, state, cost, h = batch
            if cost.sum() == 0:
                continue
            for k, v in obs.items():
                obs[k] = v.to(args.device).squeeze(1)
            labels = cost.to(args.device).squeeze(1)
            logits = fc(obs)
            if args.single_layer_classifier:
                loss = torch.nn.functional.cross_entropy(logits, labels)
            else:
                logits_safe = logits[labels == 0]
                logits_unsafe = logits[labels == 1]
                loss_safe = torch.nn.functional.relu(0.75 - logits_safe).mean() if logits_safe.shape[0] > 0 else logits.sum() * 0
                loss_unsafe = torch.nn.functional.relu(logits_unsafe + 0.75).mean() if logits_unsafe.shape[0] > 0 else logits.sum() * 0
                loss_unsafe_weak = torch.nn.functional.relu(logits_unsafe).mean() if logits_unsafe.shape[0] > 0 else logits.sum() * 0
                pred = torch.zeros_like(logits)
                pred[logits <= 0] = 1
                pred[logits > 0] = 0
                pred = pred.to(args.device).squeeze(1)
                acc = (pred == labels).float().mean().item()
                cm = confusion_matrix(labels.detach().cpu().numpy(), pred.detach().cpu().numpy())
                tn, fp, fn, tp = cm.ravel()
                h = h.to(args.device).squeeze(1)
                cosine_similarity = torch.nn.functional.cosine_similarity(h, logits.squeeze(-1), dim=-1).squeeze(-1)
                h_np = h.detach().cpu().numpy().reshape(-1)
                logits_np = logits.detach().cpu().numpy().reshape(-1)
                pearson_corr, pearson_pval = pearsonr(h_np, logits_np)
                if args.gradient_penalty:
                    obs_safe = {k:v[labels == 0] for k, v in obs.items()}
                    obs_unsafe = {k:v[labels == 1] for k, v in obs.items()}
                    z_safe = fc.encode(obs_safe)
                    z_unsafe = fc.encode(obs_unsafe)
                    N = min(z_safe.shape[0], z_unsafe.shape[0])
                    z_safe = z_safe[:N].detach()
                    z_unsafe = z_unsafe[:N].detach()
                    alpha = torch.rand(z_safe.shape[0], 1, device=args.device)
                    interpolated_z = alpha * z_safe + (1 - alpha) * z_unsafe
                    interpolated_z.requires_grad = True
                    h_interpolated = fc.head(interpolated_z)
                    gradient = torch.autograd.grad(h_interpolated, interpolated_z, grad_outputs=torch.ones_like(h_interpolated), create_graph=True, retain_graph=True, only_inputs=True)[0]
                    gradient_norm = gradient.norm(2, dim=1)
                    loss_gradient_penalty = (gradient_norm - 2.1)**2
                    loss = loss_safe.mean() + loss_unsafe.mean() + 0*loss_unsafe_weak.mean() + args.gp_weight * loss_gradient_penalty.mean()
                else:
                    loss = loss_safe.mean() + loss_unsafe.mean() + 0*loss_unsafe_weak.mean()
                pbar.set_postfix(loss=loss.item(),loss_safe=loss_safe.mean().item(),loss_unsafe=loss_unsafe.mean().item(),loss_gp=loss_gradient_penalty.mean().item() if args.gradient_penalty else 0,acc=acc, h_s=logits_safe.mean().item(), h_u=logits_unsafe.mean().item())
            train_stats['loss'].append(loss.item())
            # train_stats['loss_unsafe_weak'].append(loss_unsafe_weak.mean().item())
            if args.gradient_penalty:
                train_stats['loss_gradient_penalty'].append(loss_gradient_penalty.mean().item())
            loss.backward()
            optimizer.step()
            if not args.single_layer_classifier:
                train_stats['loss_safe'].append(loss_safe.mean().item())
                train_stats['loss_unsafe'].append(loss_unsafe.mean().item())
                logger.log({"loss/train_loss_safe":loss_safe.mean().item()})
                logger.log({"loss/train_loss_unsafe":loss_unsafe.mean().item()})
                logger.log({"acc_train/accuracy":acc})
                logger.log({"acc_train/Unsafe ACC":tp / (tp + fn)})
                logger.log({"acc_train/Safe ACC":tn / (tn + fp)})
                logger.log({"corr_train/pearson_correlation":pearson_corr})
                logger.log({"corr_train/pearson_pval":pearson_pval})
                logger.log({"corr_train/cosine_similarity":cosine_similarity.item()})
                logger.log({"cm_train/TN":tn})
                logger.log({"cm_train/FP":fp})
                logger.log({"cm_train/FN":fn})
                logger.log({"cm_train/TP":tp})
                logger.log({"h_train/h_safe_mean":logits_safe.mean().item()})
                logger.log({"h_train/h_unsafe_mean":logits_unsafe.mean().item()})
                logger.log({"h_train/h_safe_std":logits_safe.std().item()})
                logger.log({"h_train/h_unsafe_std":logits_unsafe.std().item()})
                logger.log({"h_train/h_safe_max":logits_safe.max().item()})
                logger.log({"h_train/h_unsafe_max":logits_unsafe.max().item()})
                logger.log({"h_train/h_safe_min":logits_safe.min().item()})
                logger.log({"h_train/h_unsafe_min":logits_unsafe.min().item()})
                # logger.log({"loss/train_loss_unsafe_weak":loss_unsafe_weak.mean().item()})
                if args.gradient_penalty:
                    logger.log({"loss/train_loss_gradient_penalty":loss_gradient_penalty.mean().item()})
            logger.log({"loss/train_loss":loss.item()})
            if ((idx + 1) % args.eval_freq == 0):
                stats = test(fc, val_dataloader, args, logger)
                print(f"Epoch: {_}")
                if not args.single_layer_classifier:
                    print("-----------Validation-----------")
                    print(f"Loss: {stats['loss/val_loss']:.4f}, Cosine Similarity: {stats['corr_eval/cosine_similarity']:.4f}, Pearson Correlation: {stats['corr_eval/pearson_correlation']:.4f}, Pearson P-value: {stats['corr_eval/pearson_pval']:.4f}")
                    print(f"h_safe_mean: {stats['h_eval/h_safe_mean']:.4f}, h_safe_std: {stats['h_eval/h_safe_std']:.4f}, h_safe_max: {stats['h_eval/h_safe_max']:.4f}, h_safe_min: {stats['h_eval/h_safe_min']:.4f}")
                    print(f"h_unsafe_mean: {stats['h_eval/h_unsafe_mean']:.4f}, h_unsafe_std: {stats['h_eval/h_unsafe_std']:.4f}, h_unsafe_max: {stats['h_eval/h_unsafe_max']:.4f}, h_unsafe_min: {stats['h_eval/h_unsafe_min']:.4f}")
                    print(f"Unsafe ACC: {stats['acc_eval/Unsafe ACC']:.4f}, Safe ACC: {stats['acc_eval/Safe ACC']:.4f}, Accuracy: {stats['acc_eval/accuracy']:.4f}")
                    print("--------------------------------")
                for k, v in stats.items():
                    if k not in val_stats.keys():
                        val_stats[k] = []
                    val_stats[k].append(v)
    return train_stats, val_stats

def test(fc, test_dataloader, args, logger):
    eval_stats = {}
    fc.eval()
    total_preds = torch.tensor([], dtype=torch.long, device=args.device)
    total_logits = torch.tensor([], dtype=torch.long, device=args.device)
    total_labels = torch.tensor([], dtype=torch.long, device=args.device)
    total_z = torch.tensor([], dtype=torch.float32, device=args.device)
    first_batch = True
    cosine_similarity = float("nan")
    pearson_corr, pearson_pval = float("nan"), float("nan")
    with torch.no_grad():
        for batch in test_dataloader:
            obs, act, state, cost, h = batch
            for k, v in obs.items():
                obs[k] = v.to(args.device).squeeze(1)
            labels = cost.to(args.device).squeeze(1)
            z = fc.encode(obs)
            logits = fc.head(z).squeeze(1)
            if args.single_layer_classifier:
                preds = logits.argmax(dim=-1)
            else:
                preds = torch.zeros_like(logits)
                preds[logits <= 0] = 1
                preds[logits > 0] = 0
                h = h.to(args.device).squeeze(1)
                if first_batch and (labels > 0).sum() > 0:
                    cosine_similarity = torch.nn.functional.cosine_similarity(h, logits, dim=-1).squeeze(-1)
                    h_np = h.detach().cpu().numpy().reshape(-1)
                    logits_np = logits.detach().cpu().numpy().reshape(-1)
                    pearson_corr, pearson_pval = pearsonr(h_np, logits_np)
                    first_batch = False
            total_z = torch.cat((total_z, z), dim=0)
            total_preds = torch.cat((total_preds, preds), dim=0)
            total_logits = torch.cat((total_logits, logits), dim=0)
            total_labels = torch.cat((total_labels, labels), dim=0)
        
        if args.single_layer_classifier:
            loss = torch.nn.functional.cross_entropy(total_logits, total_labels)/len(test_dataloader)
        else:
            logits_safe = total_logits[total_labels == 0]
            logits_unsafe = total_logits[total_labels == 1]
            loss_safe = torch.nn.functional.relu(0.75 - logits_safe).mean() if logits_safe.shape[0] > 0 else total_logits.sum() * 0
            loss_unsafe = torch.nn.functional.relu(logits_unsafe + 0.75).mean() if logits_unsafe.shape[0] > 0 else total_logits.sum() * 0
            loss_unsafe_weak = torch.nn.functional.relu(logits_unsafe).mean() if logits_unsafe.shape[0] > 0 else total_logits.sum() * 0
            loss = loss_safe.mean() + loss_unsafe.mean() + 0*loss_unsafe_weak.mean()
    if args.gradient_penalty:
        N = min(logits_safe.shape[0], logits_unsafe.shape[0])
        if N > 0:
            z_safe = total_z[total_labels == 0][:N].detach()
            z_unsafe = total_z[total_labels == 1][:N].detach()
            alpha = torch.rand(z_safe.shape[0], 1, device=args.device)
            interpolated_z = alpha * z_safe + (1 - alpha) * z_unsafe
            interpolated_z.requires_grad = True
            h_interpolated = fc.head(interpolated_z)
            gradient = torch.autograd.grad(h_interpolated, interpolated_z, grad_outputs=torch.ones_like(h_interpolated), create_graph=True, retain_graph=True, only_inputs=True)[0]
            gradient_norm = gradient.norm(2, dim=1)
            loss_gradient_penalty = (gradient_norm - 2.1)**2
            loss = loss_safe.mean() + loss_unsafe.mean() + 0*loss_unsafe_weak.mean() + args.gp_weight * loss_gradient_penalty.mean()
            eval_stats['loss/val_loss_gradient_penalty'] = loss_gradient_penalty.mean().item()
    eval_stats['loss/val_loss'] = loss.item()
    eval_stats['loss/val_loss_safe'] = loss_safe.mean().item()
    eval_stats['loss/val_loss_unsafe'] = loss_unsafe.mean().item()
    # eval_stats['loss/val_loss_unsafe_weak'] = loss_unsafe_weak.mean().item()
    acc = (total_preds == total_labels).float().mean().item()
    logits_safe = total_logits[total_labels == 0]
    logits_unsafe = total_logits[total_labels == 1]
    total_preds = total_preds.detach().cpu().numpy()
    total_labels = total_labels.detach().cpu().numpy()
    cm = confusion_matrix(total_labels, total_preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    def _stat(t, op):
        return op(t).item() if t.numel() > 0 else float("nan")

    eval_stats['h_eval/h_safe_mean'] = _stat(logits_safe, torch.mean)
    eval_stats['h_eval/h_unsafe_mean'] = _stat(logits_unsafe, torch.mean)
    eval_stats['h_eval/h_safe_std'] = _stat(logits_safe, torch.std)
    eval_stats['h_eval/h_unsafe_std'] = _stat(logits_unsafe, torch.std)
    eval_stats['h_eval/h_safe_max'] = _stat(logits_safe, torch.max)
    eval_stats['h_eval/h_unsafe_max'] = _stat(logits_unsafe, torch.max)
    eval_stats['h_eval/h_safe_min'] = _stat(logits_safe, torch.min)
    eval_stats['h_eval/h_unsafe_min'] = _stat(logits_unsafe, torch.min)
    eval_stats['corr_eval/cosine_similarity'] = cosine_similarity.item() if isinstance(cosine_similarity, torch.Tensor) else float("nan")
    eval_stats['corr_eval/pearson_correlation'] = pearson_corr
    eval_stats['corr_eval/pearson_pval'] = pearson_pval
    eval_stats['cm_eval/TN'] = tn
    eval_stats['cm_eval/FP'] = fp
    eval_stats['cm_eval/FN'] = fn
    eval_stats['cm_eval/TP'] = tp
    eval_stats['acc_eval/Unsafe ACC'] = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    eval_stats['acc_eval/Safe ACC'] = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    eval_stats['acc_eval/accuracy'] = acc
    logger.log(eval_stats)
    return eval_stats

def main():
    print("started")
    wandb.login()
    args = get_args_and_merge_config()
    if args.without_proprio:
        args.with_proprio = False
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    train_data, val_data, train_data_cost, _ = load_data(args.task,args.data_root,args.seed)
    label = torch.zeros(len(train_data))

    for idx, (_, _, _, cost, _) in enumerate(tqdm(train_data_cost)):
        label[idx] = cost[0]
    sum_unsafe = label.sum()
    weight_safe = 0.5/(len(train_data)-sum_unsafe)
    weight_unsafe = 0.5/sum_unsafe

    weights = torch.zeros(len(train_data))
    weights[label == 0] = weight_safe
    weights[label == 1] = weight_unsafe
    sampler_train = torch.utils.data.WeightedRandomSampler(weights=weights, num_samples=len(train_data), replacement=True)
    train_dataloader = torch.utils.data.DataLoader(
        # train_data, batch_size=args.batch_size, shuffle=True
        train_data, batch_size=args.batch_size, sampler = sampler_train, drop_last=True
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_data, batch_size=args.batch_size, shuffle=False,
    )


    # Default: all paper backbones; override with --backbones
    if args.backbones:
        backbones = args.backbones
    else:
        backbones = ["dino_cls", "vc1", "scratch", "dino", "full_scratch", "r3m", "resnet"]
    save_path = Path(args.output_root)
    for backbone in backbones:
        if "maniskill" in args.task:
            env_key = "maniskill"
        elif "carla" in args.task:
            env_key = "carla"
        elif "dubins" in args.task:
            env_key = "dubins"
        elif "cargoal" in args.task:
            env_key = "cargoal"
        else:
            env_key = None
        if env_key is not None:
            ckpt_dir = Path(args.dino_ckpt_dir) / env_key / backbone
            if backbone == "full_scratch":
                ckpt_dir = Path(args.dino_ckpt_dir) / env_key / "vc1"
        else:
            ckpt_dir = Path(args.dino_ckpt_dir) / f"{args.task}"
        hydra_cfg = ckpt_dir / 'hydra.yaml'
        snapshot = ckpt_dir / 'checkpoints' / 'model_latest.pth'

        if args.finetune:
            backbone = f"{backbone}_ft"
        training_type = "linear-probing"
        if not args.single_layer_classifier:
            training_type = "mlp_with_gp"
        config = vars(args)
        config["backbone"] = backbone
        config["ckpt_dir"] = ckpt_dir
        if args.with_proprio:
            flag = "with-proprio"
        else:
            flag = "without-proprio"
        print(config)
        if not os.path.exists(save_path / training_type / args.task  / backbone / flag):
            os.makedirs(save_path / training_type / args.task  / backbone / flag)
        logger = wandb.init(
            # Set the wandb entity where your project will be logged (generally your team name).
            entity=os.environ.get("WANDB_ENTITY", None),
            # Set the wandb project where this run will be logged.
            project=f"{training_type}-{flag}-{args.task}",
            dir= save_path / training_type / args.task  / backbone / flag ,
            # Track hyperparameters and run metadata.
            config=config,
            name=f"{backbone}"
        )
        # load train config and model weights
        train_cfg = OmegaConf.load(str(hydra_cfg))
        num_action_repeat = train_cfg.num_action_repeat
        wm = load_model(snapshot, train_cfg, num_action_repeat, device=args.device)
        wm.eval()
        if backbone == "full_scratch":
            for m in wm.modules():
                if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):
                    torch.nn.init.xavier_uniform_(m.weight)
            for p in wm.parameters():
                p.requires_grad = True
        if args.finetune or (backbone == "full_scratch"):
            print("freeze_wm: False")
            args.freeze_wm = False
        else:
            print("freeze_wm: True")
            args.freeze_wm = True

        obs, _, _, _, _ = next(iter(val_dataloader))
        for k, v in obs.items():
            obs[k] = v.to(args.device).squeeze(1)
        fc = FailureClassifier(obs, wm, args).to(args.device)
        train_stats, val_stats = train(fc, train_dataloader, val_dataloader, args, logger)
        eval_stats = test(fc, val_dataloader, args, logger)

        # ckpts and stats will be saved to the original ckpt_dir
        torch.save(
            {
                'model_state_dict': fc.state_dict(),
            },
            save_path / training_type / args.task  / backbone / flag / 'failure_classifier.pth'
        )

        torch.save({
            'train_stats': train_stats,
            'val_stats': val_stats,
            'eval_stats': eval_stats
        }, save_path / training_type / args.task  / backbone / flag / 'stats.pth')

        wandb.finish()

if __name__ == "__main__":
    main()

    # python train_failure_classifier.py --task maniskillnew

    # train_dataloader = torch.utils.data.DataLoader(
    #     # train_data, batch_size=args.batch_size, shuffle=True
    #     train_data, batch_size=64, sampler = sampler_train
    # )
    # _,_,_,cost=next(iter(train_dataloader))