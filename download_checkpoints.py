"""Download the pretrained checkpoints from Hugging Face and arrange them
into the local layout expected by the training/eval scripts.

Usage:
    python download_checkpoints.py            # symlink into checkpoints/ (default)
    python download_checkpoints.py --copy     # copy instead of symlink (needs ~27 GB extra)

Repo: https://huggingface.co/JsSparkYyx/LatentSafetyFilterWithPVRs
"""
import argparse
import os
from pathlib import Path

from huggingface_hub import snapshot_download

REPO_ID = "JsSparkYyx/LatentSafetyFilterWithPVRs"
ROOT = Path(__file__).resolve().parent
SNAPSHOT_DIR = ROOT / "checkpoints" / "_hf_snapshot"

# HF path prefix -> local path under checkpoints/
# dinowm/<env>/<bb>/{model_latest.pth,hydra.yaml}
#   -> checkpoints/wm/<env>/<bb>/{checkpoints/model_latest.pth,hydra.yaml}
# hj/ddpg_hj_latent_dubins/<bb>/*    -> checkpoints/hj/dubins/<bb>/epoch_200/*
# hj/ddpg_hj_latent_cargoal/<bb>/*   -> checkpoints/hj/cargoal/<bb>/latest/*
# hj/ddpg_hj_latent_mani_mlp_h/<bb>/* -> checkpoints/hj/maniskill/<bb>/latest/*
HJ_MAP = {
    "ddpg_hj_latent_dubins": ("dubins", "epoch_200"),
    "ddpg_hj_latent_cargoal": ("cargoal", "latest"),
    "ddpg_hj_latent_mani_mlp_h": ("maniskill", "latest"),
}


def local_path(hf_path: str) -> Path:
    parts = hf_path.split("/")
    if parts[0] == "dinowm":
        _, env, bb, fname = parts
        if fname == "model_latest.pth":
            return ROOT / "checkpoints" / "wm" / env / bb / "checkpoints" / fname
        return ROOT / "checkpoints" / "wm" / env / bb / fname
    if parts[0] == "hj":
        _, run, bb, fname = parts
        env, epoch_dir = HJ_MAP[run]
        return ROOT / "checkpoints" / "hj" / env / bb / epoch_dir / fname
    if parts[0] == "nominal":
        _, fname = parts
        if fname == "reaching.pt":
            return ROOT / "nominal_controller" / fname
        return ROOT / "checkpoints" / fname
    raise ValueError(f"unexpected HF path: {hf_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--copy", action="store_true",
                        help="copy files instead of creating symlinks")
    args = parser.parse_args()

    snapshot = snapshot_download(REPO_ID, local_dir=str(SNAPSHOT_DIR))
    snapshot = Path(snapshot)

    n = 0
    for hf_file in sorted(snapshot.rglob("*")):
        if not hf_file.is_file():
            continue
        rel = hf_file.relative_to(snapshot).as_posix()
        if rel.startswith("."):  # .cache/ metadata from snapshot_download
            continue
        dest = local_path(rel)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() or dest.is_symlink():
            dest.unlink()
        if args.copy:
            import shutil
            shutil.copy2(hf_file, dest)
        else:
            dest.symlink_to(hf_file)
        n += 1
    print(f"Linked {n} files into {ROOT / 'checkpoints'}")
    if not args.copy:
        print(f"Note: symlinks point into {SNAPSHOT_DIR}; do not delete it "
              f"(or re-run with --copy).")


if __name__ == "__main__":
    main()
