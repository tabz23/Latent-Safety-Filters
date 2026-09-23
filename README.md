# Official Code for the Paper Designing Latent Safety Filters using Pre-Trained Vision Models
# Webpage: https://trustworthyautonomy.github.io/LatentSafetyFilterWithPVRs/



**Designing Latent Safety Filters using Pre-Trained Vision Models**<br>
Ihab Tabbara\*, Yuxuan Yang\*, Ahmad Hamzeh, Maxwell Astafyev, Hussein Sibai<br>
Washington University in St. Louis — [arXiv:2509.14758](https://arxiv.org/abs/2509.14758)
Paper

We systematically evaluate pre-trained vision models (PVRs) — DINOv2 (patch
embeddings and CLS token), VC-1, R3M and ResNet-50 — as perception backbones
for vision-based safety filters: latent failure classifiers and
Hamilton–Jacobi (HJ) reachability-based safety filters trained with
DDPG/DDQN, on top of latent world models adapted from
[DINO-WM](https://github.com/gaoyuezhou/dino_wm). Experiments cover four
environments: Dubins car, Safety-Gymnasium CarGoal, ManiSkill
(`UnitreeG1PlaceAppleInBowl`) and CARLA.

The code is organized in three parts:

1. **Training latent world models** (DINO-WM with any of the PVR backbones)
2. **Training safety filters** (latent failure classifier + HJ value function)
3. **Testing safety filters** (closed-loop evaluation with critic-only or
   dynamics-lookahead switching, saliency maps, result aggregation)

Pretrained checkpoints for all environments and backbones are hosted at
<https://huggingface.co/JsSparkYyx/LatentSafetyFilterWithPVRs>.

## Repository layout

```
train.py                          # Part 1: DINO-WM training (hydra)
conf/                             # hydra configs (train_latent_safety.yaml is the paper config)
models/                           # world model + PVR encoders (DINOv2, VC-1, R3M, ResNet)
datasets/                         # trajectory datasets for the 4 environments
env/                              # environment wrappers + dataset generation scripts

train_failure_classifier.py       # Part 2a: latent failure classifier (hinge loss + gradient penalty)
train_HJ_dubinslatent_withfinetune_ddqn.py   # Part 2b: HJ filter, Dubins (DDQN)
train_HJ_dubinslatent_withfinetune.py        # Part 2b: HJ filter, Dubins (DDPG) note that DDPG does not work well. Use DDQN instead for dubins.
train_HJ_visual_ft.py                        # Part 2b: HJ filter, CarGoal (DDPG)
train_HJ_mani_ft.py                          # Part 2b: HJ filter, ManiSkill (DDPG, uses latent h)
train_HJ_dubinstruth{,_BRS}.py               # ground-truth-state Dubins baseline (needs PyHJ, see below)

test_dubins_latent_specific_checkpoint.py    # Part 3: closed-loop eval, Dubins
test_cargoal_latent_specific_checkpoint.py   # Part 3: closed-loop eval, CarGoal
test_maniskill_latent_specific_checkpoint.py # Part 3: closed-loop eval, ManiSkill
saliency_new_{dubins,cargoal,mani}.py        # occlusion-based saliency maps of the HJ value
gen_csv.py, plots.py                         # aggregate closed-loop summaries / paper plots

nominal_controller/               # vendored DreamerV3 nominal policy for CarGoal (code + ckpt)
checkpoints/maniskill_ppo_expert.pt          # vendored PPO nominal policy for ManiSkill
plan.py, planning/, utils.py, preprocessor.py, custom_resolvers.py, metrics/, distributed_fn/
download_checkpoints.py           # fetch pretrained ckpts from Hugging Face
```

## Installation

Two conda environments are used: a main one (Dubins, CarGoal, CARLA) and one
for ManiSkill.

### Main environment (Dubins / CarGoal / CARLA)

```bash
conda create -n latentsf python=3.9 -y
conda activate latentsf
pip install torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# VC-1 backbone (install from source):
git clone https://github.com/facebookresearch/eai-vc.git
pip install -e ./eai-vc/vc_models

# CarGoal environment (install from source):
git clone https://github.com/PKU-Alignment/safety-gymnasium.git
pip install -e ./safety-gymnasium
```

### ManiSkill environment

```bash
conda create -n latentsf_mani python=3.9 -y
conda activate latentsf_mani
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements_maniskill.txt
```

ManiSkill 3 (SAPIEN) requires an EGL/Vulkan-capable system for rendering. We
run all ManiSkill jobs inside the `maniskill/base` Docker image (on SLURM via
pyxis: `--container-image=maniskill/base`); the conda env above supplies the
Python packages inside that image. On clusters without container support, a
bare GPU node with working EGL drivers also works.

### Notes

- **Backbone weights** are downloaded automatically on first use: DINOv2 via
  `torch.hub` (honors `TORCH_HOME`), R3M via `gdown` (honors `R3M_CKPT_DIR`,
  default `~/.cache/r3m`), VC-1 via the `eai-vc` package.
- **Weights & Biases**: training scripts log to wandb. Set `WANDB_ENTITY` to
  your entity, or `WANDB_MODE=disabled` to turn logging off.
- **PyHJ** is only needed for the optional ground-truth Dubins baseline
  (`train_HJ_dubinstruth*.py`): install the Tianshou fork from
  <https://github.com/jamesjingqili/Lipschitz_Continuous_Reachability_Learning>.
  All latent-space safety-filter training scripts are self-contained and do
  not need it.
- The `environment.yaml` / `requirements_pasted.txt` files from the upstream
  DINO-WM repo are superseded by the two requirements files above.

## Checkpoints

Download all pretrained world models and safety filters (~26 GB):

```bash
python download_checkpoints.py          # symlinks into checkpoints/
python download_checkpoints.py --copy   # full local copy instead
```

This arranges the Hugging Face files into the layout the scripts expect:

```
checkpoints/
  wm/<env>/<backbone>/hydra.yaml                       # env ∈ {dubins,cargoal,maniskill,carla}
  wm/<env>/<backbone>/checkpoints/model_latest.pth     # backbone ∈ {dino,dino_cls,r3m,resnet,scratch,vc1}
  hj/dubins/<backbone>[_ft]/epoch_200/critic.pth       # Dubins DDQN critics
  hj/cargoal/<backbone>[_ft]/latest/{actor,critic,wm}.pth
  hj/maniskill/<backbone>[_ft]/latest/{actor,critic,wm}.pth
  maniskill_ppo_expert.pt                              # ManiSkill PPO nominal policy
nominal_controller/reaching.pt                         # CarGoal Dreamer nominal policy
```

`_ft` variants were trained with backbone fine-tuning; their `wm.pth` holds
the fine-tuned world model. `full_scratch` is the from-scratch ViT variant.
The latent failure classifiers are *not* hosted — train them with Part 2a
below (needed only to reproduce ManiSkill HJ training, which uses the learned
classifier as its signed distance).

## Datasets

Datasets are not hosted; collect them by running (from the repo root):

```bash
# Dubins: 1800 PID trajectories -> datasets/dubins1800_continuous_cost
python env/dubins/generate_dataset.py

# CarGoal: 2000 episodes with the vendored Dreamer nominal policy
python env/cargoal/generate_dataset.py --num_episodes 2000 --output_dir datasets/cargoalnewshort

# ManiSkill: 2000 episodes (world-model data) and 3000 episodes with binary
# failure labels (failure-classifier data), using the PPO nominal policy
python env/maniskill/maniskill_generatedata.py --num-episodes 2000 --output-dir datasets/maniskillnew
python env/maniskill/maniskill_generatedata_classifier.py --num-episodes 3000 --output-dir datasets/maniskill3000classif

# CARLA: 2000 trajectories (requires a running CARLA simulator, Town10)
python env/carla/generate_dataset.py
```

Each dataset directory contains `states.pth`, `actions.pth`,
`seq_lengths.pth`, `costs.pth` (lists of per-episode tensors) and
`obses/episode_XXX.pth` (per-episode `[T, 224, 224, 3]` uint8 frames).
Training configs read the dataset root from the `DATASET_DIR` environment
variable (e.g. `export DATASET_DIR=$PWD/datasets`), or override
`env.dataset.data_path` directly.

## Part 1 — Train DINO-WM

```bash
python train.py --config-name train_latent_safety.yaml \
  env=dubins frameskip=1 num_hist=3 encoder=dino_cls decoder=transposed_conv \
  proprio_emb_dim=10 training.batch_size=32 training.epochs=100
```

- `encoder` ∈ `dino` (patch tokens, uses `decoder=vqvae`), `dino_cls`, `vc1`,
  `r3m`, `resnet`; `encoder=scratch model.train_encoder=true` trains the
  ResNet representation end-to-end with the dynamics (the "WM-R" variant).
- `env` ∈ `dubins`, `cargoal`, `maniskill`, `carla` (see `conf/env/*.yaml`;
  paper settings: `proprio_emb_dim` 10/40/50/20 respectively, `frameskip=1`,
  `num_hist=3`, 100 epochs).
- Checkpoints are written to `${ckpt_base_path}/outputs/<date>/<time>/`
  (`checkpoints/` + `hydra.yaml`).

## Part 2 — Train safety filters

### 2a. Latent failure classifier

```bash
python train_failure_classifier.py --seed 1 --task dubins1800_continuous_cost \
  --data_root datasets --dino_ckpt_dir checkpoints/wm \
  --output_root checkpoints/classifier --epochs 10            # add --finetune for the FT variant
```

`--task` selects the dataset (`dubins1800_withcost`, `dubins1800_continuous_cost`,
`cargoalnewshort`, `maniskillnew`, `maniskill3000classif`, `carla_distance_cost`,
`carla_2k_v`); `--backbones` restricts the backbone sweep (default: all).
The classifier MLP is trained with a hinge loss plus a gradient penalty on
latent interpolations, and logs accuracy and correlation with the ground-truth
distance. For ManiSkill (binary failure labels only), the trained classifier
serves as the signed distance for HJ training.

### 2b. HJ value function

```bash
# Dubins (DDQN, 3 discrete actions)
python train_HJ_dubinslatent_withfinetune_ddqn.py \
  --dino_ckpt_dir checkpoints/wm/dubins --config train_HJ_configs.yaml \
  --dino_encoder dino_cls --total-episodes 200 --step-per-epoch 200 \
  --batch_size-pyhj 64 --gamma-pyhj 0.99 --critic-net 512 512 512
  # add --with_finetune --encoder_lr 1e-6 for the FT variant

# CarGoal (DDPG; nominal/exploration policy = vendored Dreamer)
python train_HJ_visual_ft.py --config train_HJ_visual_configs.yaml \
  --dino_ckpt_dir checkpoints/wm/cargoal --dino_encoder dino_cls

# ManiSkill (DDPG; signed distance = learned classifier from Part 2a)
python train_HJ_mani_ft.py --config train_HJ_mani_configs.yaml \
  --dino_ckpt_dir checkpoints/wm/maniskill \
  --latent_h_ckpt checkpoints/classifier/mlp_with_gp/maniskill3000classif \
  --dino_encoder dino_cls --use_latent_h \
  --expert_ckpt checkpoints/maniskill_ppo_expert.pt
```

All support `--dino_encoder {dino,dino_cls,vc1,r3m,resnet,scratch,full_scratch}`
and `--with_finetune`. Runs are logged under `runs/`.

## Part 3 — Closed-loop evaluation

```bash
# Dubins: critic-only switching, 50 episodes
python test_dubins_latent_specific_checkpoint.py --backbone dino_cls --num_runs 50 \
  --wm_ckpt_root checkpoints/wm/dubins --hj_ckpt_dir checkpoints/hj/dubins \
  --output_dir close_loop
# add --only_dynamics for dynamics-lookahead switching, --pid_only for the
# no-filter baseline, --finetune for FT variants, --eps <tau> for the margin

# CarGoal (nominal = vendored Dreamer policy)
python test_cargoal_latent_specific_checkpoint.py --backbone dino_cls --num_runs 50 \
  --wm_ckpt_root checkpoints/wm/cargoal --hj_ckpt_dir checkpoints/hj/cargoal \
  --output_dir close_loop

# ManiSkill (nominal = vendored PPO policy)
python test_maniskill_latent_specific_checkpoint.py --backbone dino_cls --num_runs 50 \
  --wm_ckpt_root checkpoints/wm/maniskill --hj_ckpt_dir checkpoints/hj/maniskill \
  --output_dir close_loop
```

Each run writes `summary.txt` (steps, violations, min HJ, success rate,
interventions, inference timings) plus rollout videos and switching-debug
plots under `close_loop/<env>_<eps>/{critic,dynamics,pid}/<backbone>/`.
Aggregate with `python gen_csv.py --results_dir close_loop` and plot with
`plots.py`.

Saliency maps (occlusion-based, on the HJ value):

```bash
python saliency_new_dubins.py --task dubins --data_path datasets --backbone dino_cls \
  --wm_ckpt_root checkpoints/wm/dubins --hj_ckpt_dir checkpoints/hj/dubins \
  --occlusion_method black --num_samples 20 --output_dir outputs/saliency
```

## Caveats

- CARLA is supported for world-model and failure-classifier training
  (correlation results in the paper); no CARLA HJ training or closed-loop
  evaluation code is included.
- ManiSkill: creating a fresh SAPIEN env after running encoder inference can
  segfault on some GPU/driver setups; the eval script creates its env before
  any inference for this reason.

## Acknowledgements

Built on [DINO-WM](https://github.com/gaoyuezhou/dino_wm) (NYU),
[dreamerv3-torch](https://github.com/NM512/dreamerv3-torch),
[R3M](https://github.com/facebookresearch/r3m),
[VC-1/eai-vc](https://github.com/facebookresearch/eai-vc),
[DINOv2](https://github.com/facebookresearch/dinov2),
[ManiSkill](https://github.com/haosulab/ManiSkill),
[Safety-Gymnasium](https://github.com/PKU-Alignment/safety-gymnasium) and
[CARLA](https://carla.org/).

## Citation

```bibtex
@article{tabbara2025latentsafety,
  title={Designing Latent Safety Filters using Pre-Trained Vision Models},
  author={Tabbara, Ihab and Yang, Yuxuan and Hamzeh, Ahmad and Astafyev, Maxwell and Sibai, Hussein},
  journal={arXiv preprint arXiv:2509.14758},
  year={2025}
}
```
