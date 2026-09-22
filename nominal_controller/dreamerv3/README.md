# dreamerv3-torch
Pytorch implementation of [Mastering Diverse Domains through World Models](https://arxiv.org/abs/2301.04104v1). DreamerV3 is a scalable algorithm that outperforms previous approaches across various domains with fixed hyperparameters.

## Instructions

### Method 1: Manual

Get dependencies with python 3.11:
```
pip install -r requirements.txt
```
Run training on DMC Vision:
```
python dreamer.py --configs dmc_vision --task dmc_walker_walk --logdir ./logdir/dmc_walker_walk
```
Monitor results:
```
tensorboard --logdir ./logdir
```
To set up Atari or Minecraft environments, please check the scripts located in [env/setup_scripts](https://github.com/NM512/dreamerv3-torch/tree/main/envs/setup_scripts).

Train Dreamer for Goal Reaching:
```
python dreamer.py --configs safe_gymnasium --task sg_SafetyCarGoal2Vision-v0 --logdir ./logdir/SafetyCarGoal2128-v0
```
Train Dreamer for Cost Reduction:
```
python dreamer.py --configs safe_gymnasium --task sgc_SafetyCarGoal2Vision-v0 --logdir ./logdir/SafetyCarGoal2Cost128-v0
```
