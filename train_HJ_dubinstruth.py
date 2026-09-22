import argparse
import os
from pathlib import Path
import torch
import gymnasium as gym  # We use Gymnasium API for vector envs
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from omegaconf import OmegaConf

# PyHJ components for DDPG training
from PyHJ.data import Collector, VectorReplayBuffer, Batch
from PyHJ.trainer import offpolicy_trainer
from PyHJ.env import DummyVectorEnv
from PyHJ.exploration import GaussianNoise
from PyHJ.utils import TensorboardLogger, WandbLogger
from PyHJ.utils.net.common import Net
from PyHJ.utils.net.continuous import Actor, Critic
from PyHJ.policy import avoid_DDPGPolicy_annealing
# Load your DINO-WM via plan.load_model
from plan import load_model

# Underlying Dubins Gym env (classic Gym)
from env.dubins.dubins import DubinsEnv
from gymnasium.spaces import Box

import yaml

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
    parser = argparse.ArgumentParser("DDPG HJ on DINO latent Dubins")
    parser.add_argument(
        "--dino_ckpt_dir", type=str,
        default="checkpoints/wm/dubins",
        help="Where to find the DINO-WM checkpoints"
    )
    parser.add_argument(
        "--config", type=str, default="train_HJ_configs.yaml",
        help="Path to your flat YAML of hyperparameters"
    )
    parser.add_argument(
        "--gamma-pyhj", type=float, default=None,
        help="(Optional) override gamma_pyhj from the config"
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
        setattr(args, key.replace("-", "_"), val)

    # 5) Respect any explicit CLI override of --gamma-pyhj
    if args.gamma_pyhj is not None:
        args.gamma_pyhj = args.gamma_pyhj

    return args


class DubinsEnvWrapper(gym.Env):
    """
    Wraps the classic Gym-based DubinsEnv into a Gymnasium-compatible Env,
    but *only* returns the proprioceptive (x,y,theta) state instead of the DINO latent.
    """

    def __init__(self):
        super().__init__()
        self.env = DubinsEnv()
        # Proprio space is already defined on the base env:
        self.observation_space = self.env.proprio_space   # Box(low=[-3,-3,-π], high=[3,3,π])
        self.action_space      = self.env.action_space

    def reset(self):
        state = self.env.reset()[0]     # obs_dict
        proprio = state["proprio"]      # np.array([x,y,θ])
        return proprio, {}             # now obs is (3,)

    def step(self, action):
        obs_dict, reward, done, info = self.env.step(action)
        proprio = obs_dict["proprio"]
        # Use h‐value as reward if you still want:
        h_s = info.get("h", 0.0) * 3
        return proprio, h_s, done, False, info

    # You can drop _encode entirely since we’re not using DINO.

import os
# point Matplotlib to /tmp (or any other writable dir)
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
# make sure it exists
os.makedirs(os.environ['MPLCONFIGDIR'], exist_ok=True)

# now it’s safe to import pyplot
import matplotlib.pyplot as plt
import wandb

# … all your existing imports …

from PyHJ.data import Batch

def compute_hj_value(x: float, y: float, theta: float,
                     policy, args) -> float:
    """
    Compute the learned Q(s, π_old(s)) at the given (x, y, theta).
      1) Build the raw proprio state s = [x,y,theta]
      2) Wrap into a Batch so policy API is happy
      3) Query old‐actor for action and critic for Q‐value
      4) Return Q
    """
    # 1) raw state vector
    # state = np.array([x, y, theta], dtype=np.float32)
    # create a batch of size 1
    state = np.array([x, y, theta], dtype=np.float32)[None, ...]  # (1, 3)
    # 2) Batch wrapper
    batch = Batch(obs=state, info=Batch())
    # 3) actor_old(s) → a_old; critic(s, a_old) → Q
    a_old = policy(batch, model="actor_old").act
    q_val = policy.critic(batch.obs, a_old).cpu().item()
    return q_val



def plot_hj(policy, thetas, args):
    """
    Plot only the learned Q‐value over an (x,y) grid for each heading in `thetas`.
    """
    xs = np.linspace(args.x_min, args.x_max, args.nx)
    ys = np.linspace(args.y_min, args.y_max, args.ny)
    fig1, axes1 = plt.subplots(len(thetas), 1, figsize=(3, 3*len(thetas)))
    fig2, axes2 = plt.subplots(len(thetas), 1, figsize=(3, 3*len(thetas)))

    for i, theta in enumerate(thetas):
        vals = np.zeros((args.nx, args.ny), dtype=np.float32)
        for ix, x in enumerate(xs):
            for iy, y in enumerate(ys):
                vals[ix, iy] = compute_hj_value(x, y, theta, policy, args)

        # Binary safe/unsafe: Q > 0
        axes1[i].imshow(
            (vals.T > 0),
            extent=(args.x_min, args.x_max, args.y_min, args.y_max),
            origin="lower"
        )
        axes1[i].set_title(f"θ={theta:.2f} (Q>0)")

        # Continuous Q‐value plot
        im = axes2[i].imshow(
            vals.T,
            extent=(args.x_min, args.x_max, args.y_min, args.y_max),
            origin="lower"
        )
        axes2[i].set_title(f"θ={theta:.2f} (Q)")
        fig2.colorbar(im, ax=axes2[i])

    fig1.tight_layout()
    fig2.tight_layout()
    return fig1, fig2

def main():
    args = get_args_and_merge_config()
    # cast hyperparams to proper types
    args.critic_lr         = float(args.critic_lr)
    args.actor_lr          = float(args.actor_lr)
    args.tau               = float(args.tau)
    args.gamma_pyhj        = float(args.gamma_pyhj)
    args.exploration_noise = float(args.exploration_noise)
    args.update_per_step   = float(args.update_per_step)
    args.step_per_epoch    = int(args.step_per_epoch)
    args.step_per_collect  = int(args.step_per_collect)
    args.test_num          = int(args.test_num)
    args.training_num      = int(args.training_num)
    args.total_episodes    = int(args.total_episodes)
    args.batch_size_pyhj   = int(args.batch_size_pyhj)
    args.buffer_size       = int(args.buffer_size)
    
    # random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)            # if you use CUDA
    # torch.backends.cudnn.deterministic = True        # ▸ slower, deterministic
    torch.backends.cudnn.benchmark     = False
    
    # 1) Initialize W&B and TensorBoard writer
    # 1) Generate a unique run name, initialize W&B and TensorBoard
    import wandb
    from datetime import datetime
    run_name = datetime.now().strftime("%Y%m%d-%H%M%S")
    base_ckpt_dir = Path("runs/ddpg_hj_dubins")
    wandb.init(
        project="ddpg-hj-dubins",
        name=run_name,
        config=vars(args)
    )
    writer   = SummaryWriter(log_dir=f"runs/ddpg_hj_dino/{run_name}")
    wb_logger = WandbLogger()
    wb_logger.load(writer)                   # <-- this is required
    logger   = wb_logger                     # use only W&B logger

    # 2) Create envs
    train_envs = DummyVectorEnv([lambda: DubinsEnvWrapper() for _ in range(args.training_num)])
    test_envs  = DummyVectorEnv([lambda: DubinsEnvWrapper() for _ in range(args.test_num)])

    # 3) Shapes & max_action
    state_space  = train_envs.observation_space[0]
    action_space = train_envs.action_space[0]
    state_shape  = state_space.shape
    action_shape = action_space.shape or action_space.n
    max_action   = torch.tensor(action_space.high, device=args.device, dtype=torch.float32)

    # 4) Build critic & actor
    critic_net = Net(
        state_shape, action_shape,
        hidden_sizes=args.critic_net,
        activation=getattr(torch.nn, args.critic_activation),
        concat=True, device=args.device
    )
    critic       = Critic(critic_net, device=args.device).to(args.device)
    critic_optim = torch.optim.AdamW(
        critic.parameters(), lr=args.critic_lr,
        weight_decay=args.weight_decay_pyhj
    )

    actor_net = Net(
        state_shape,
        hidden_sizes=args.control_net,
        activation=getattr(torch.nn, args.actor_activation),
        device=args.device
    )
    actor       = Actor(actor_net, action_shape, max_action=max_action, device=args.device).to(args.device)
    actor_optim = torch.optim.AdamW(actor.parameters(), lr=args.actor_lr)

    # 5) Assemble policy
    policy = avoid_DDPGPolicy_annealing(
        critic=critic, critic_optim=critic_optim,
        tau=args.tau, gamma=args.gamma_pyhj,
        exploration_noise=GaussianNoise(sigma=args.exploration_noise),
        reward_normalization=args.rew_norm,
        estimation_step=args.n_step,
        action_space=action_space,
        actor=actor, actor_optim=actor_optim,
        actor_gradient_steps=args.actor_gradient_steps,
    )

    # 6) Collectors
    buffer          = VectorReplayBuffer(args.buffer_size, args.training_num)
    train_collector = Collector(policy, train_envs, buffer, exploration_noise=True)
    print("collecting some data first")
    train_collector.collect(10000)
    print("done collecting some data first")
    test_collector  = Collector(policy, test_envs)

    # 7) Precompute HJ slices on (x,y) for a few headings
    thetas = [0.0, np.pi/4, np.pi/2, 3*np.pi/4]
    helper_env = DubinsEnv()  # for raw [x,y,θ] resets

    # 8) Training loop, one epoch at a time
    # 8) Use the run_name to differentiate checkpoints
    log_path = base_ckpt_dir / run_name
    log_path.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.total_episodes + 1):
        print(f"\n=== Epoch {epoch}/{args.total_episodes} ===")

        # a) one epoch of off-policy training
        stats = offpolicy_trainer(
            policy=policy,
            train_collector=train_collector,
            test_collector=test_collector,
            max_epoch=1,
            step_per_epoch=args.step_per_epoch,
            step_per_collect=args.step_per_collect,
            episode_per_test=args.test_num,
            batch_size=args.batch_size_pyhj,
            update_per_step=args.update_per_step,
            stop_fn=lambda r: False,
            save_best_fn=None,
            logger=logger,
        )
        # log training metrics to W&B
        # wandb.log({f"train/{k}": v for k, v in stats.items()}, step=epoch)
        # keep only numeric scalars
        to_log = {}
        for k, v in stats.items():
            # if it’s already a Python float/int, keep it
            if isinstance(v, (float, int)):
                to_log[f"train/{k}"] = v
            # if it's a numpy scalar, cast to float
            elif isinstance(v, np.generic):
                to_log[f"train/{k}"] = float(v)
            # otherwise skip (e.g. strings like '0.12s' or arrays)
        # now log only numeric metrics
        wandb.log(to_log, step=epoch)


        # b) save policy checkpoint under this run’s subdir
        ckpt_dir = log_path / f"epoch_{epoch:03d}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save(policy.state_dict(), ckpt_dir / "policy.pth")

        # c) plot HJ slices and send to W&B
        fig1, fig2 = plot_hj(policy, thetas, args)
        wandb.log({"HJ/binary": wandb.Image(fig1),
                   "HJ/continuous": wandb.Image(fig2)}, step=epoch)
        plt.close(fig1)
        plt.close(fig2)

    print("Training complete.")

if __name__ == "__main__":
    main()
