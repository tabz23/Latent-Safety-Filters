#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Iterable
import csv

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import numpy as np
from brokenaxes import brokenaxes
from statistics import mean


BASE = "#B8DBB3"   # hollow edges for base
FT   = "#E29135" 

def _lims_with_padding(vals, lo_pad_frac=0.05, hi_pad_frac=0.10,
                       lo_clip=None, hi_clip=None, min_span=1e-3):
    """Return (lo, hi) padded around data by a fraction of its span."""
    vmin = min(vals); vmax = max(vals)
    span = max(vmax - vmin, min_span)
    lo = vmin - lo_pad_frac * span
    hi = vmax + hi_pad_frac * span
    if lo_clip is not None:
        lo = max(lo_clip, lo)
    if hi_clip is not None:
        hi = min(hi_clip, hi)
    # guard tiny or inverted ranges
    if hi - lo < min_span:
        mid = 0.5 * (lo + hi)
        half = 0.5 * max(min_span, span * 0.05)
        lo, hi = mid - half, mid + half
    return lo, hi

def _to_float_or_none(x):
    if x is None: return None
    s = str(x).strip()
    if s == "" or s.lower() == "none":
        return None
    try:
        return float(s)
    except Exception:
        return None

def _maybe_float(x):
    if x is None:
        return None
    s = str(x).strip()
    if s == "" or s.lower() == "none":
        return None
    return float(s)

def _to_bool(s):
    if s is None: return False
    s = str(s).strip().lower()
    return s in ("1", "true", "yes")

def _read_csv(path: Path) -> List[Dict[str, Any]]:
    import csv
    rows = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader, start=1):
            try:
                rows.append({
                    "env": str(row["env"]).strip(),
                    "backbone": str(row["backbone"]).strip(),
                    "safe_acc": float(row["safe_acc"]),
                    "unsafe_acc": float(row["unsafe_acc"]),
                    "avg_violation": _maybe_float(row.get("avg_violation")),
                    "avg_success": _maybe_float(row.get("avg_success")),
                })
            except KeyError as e:
                raise ValueError(f"CSV missing required column {e}. Required: env, backbone, safe_acc, unsafe_acc") from e
            except Exception as e:
                raise ValueError(f"Bad row #{i}: {row}. {e}") from e
    return rows

def _read_json(path: Path) -> List[Dict[str, Any]]:
    # Accept either a list of dicts, or a dict with a top-level list under "data"
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, list):
        items = obj
    elif isinstance(obj, dict) and "data" in obj and isinstance(obj["data"], list):
        items = obj["data"]
    else:
        raise ValueError("JSON must be a list of {env, backbone, safe_acc, unsafe_acc} or {'data': [...]}")
    rows = []
    for i, row in enumerate(items, start=1):
        try:
            rows.append({
                "env": str(row["env"]).strip(),
                "backbone": str(row["backbone"]).strip(),
                "safe_acc": float(row["safe_acc"]),
                "unsafe_acc": float(row["unsafe_acc"]),
                "avg_violation": _maybe_float(row.get("avg_violation")),
                "avg_success": _maybe_float(row.get("avg_success")),
            })
        except KeyError as e:
            raise ValueError(f"JSON item missing required field {e} at index {i-1}. Required: env, backbone, safe_acc, unsafe_acc") from e
        except Exception as e:
            raise ValueError(f"Bad item #{i}: {row}. {e}") from e
    return rows

def read_input(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Input not found: {path}")
    if path.suffix.lower() == ".csv":
        return _read_csv(path)
    if path.suffix.lower() == ".json":
        return _read_json(path)
    raise ValueError("Unsupported input format. Use .csv or .json")

def validate_rows(rows: List[Dict[str, Any]]) -> None:
    for i, r in enumerate(rows, start=1):
        for k in ("safe_acc", "unsafe_acc"):
            v = r[k]
            if not (0.0 <= v <= 1.0):
                raise ValueError(f"Row #{i} has {k}={v} not in [0,1]: {r}")

def group_by_env(rows: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    env_map: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        env_map.setdefault(r["env"], []).append(r)
    return env_map

def clean_backbone_name(name):
        return name[:-3] if name.endswith("_ft") else name

def make_scatter_for_env(env: str,
                         items: List[Dict[str, Any]],
                         outdir: Path,
                         fmt: str,
                         annotate: bool,
                         tight_layout: bool,
                         title_suffix: Optional[str],
                         label_fmt: str) -> Path:
    # Sort for consistent plotting order (optional)
    items = sorted(items, key=lambda r: (-(r["safe_acc"] + r["unsafe_acc"]), r["backbone"]))
    xs_ft = [r["safe_acc"] for r in items if r["backbone"].endswith("_ft")]
    ys_ft = [r["unsafe_acc"] for r in items if r["backbone"].endswith("_ft")]
    labels_ft = [clean_backbone_name(r["backbone"]) for r in items if r["backbone"].endswith("_ft")]

    xs_base = [r["safe_acc"] for r in items if not r["backbone"].endswith("_ft")]
    ys_base = [r["unsafe_acc"] for r in items if not r["backbone"].endswith("_ft")]
    labels_base = [f"{r['backbone']}" for r in items if not r["backbone"].endswith("_ft")]

    fig = plt.figure(figsize=(8,8))
    use_broken_x = False
    xs = xs_ft + xs_base
    ys = ys_ft + ys_base
    xmin = max(0.0, min(xs) - 0.02)
    xmax = min(1.0, max(xs) + 0.15)
    ymin = max(0.0, min(ys) - 0.02)
    ymax = min(1.0, max(ys) + 0.02)

    if use_broken_x:
        # Example: two x windows that skip the middle
        # (left window around 0.85–0.93, right window around 0.97–xmax)
        # Adjust these to match your data cluster(s)
        left_lo  = max(0.0, xmin)
        left_hi  = min(0.93, xmax)
        right_lo = max(0.97, xmin)
        right_hi = xmax

        # Ensure the windows don’t overlap and are within [0,1]
        if right_lo <= left_hi:
            right_lo = min( right_hi, left_hi + 0.01 )

        bax = brokenaxes(
            xlims=((left_lo, left_hi), (right_lo, right_hi)),
            ylims=((ymin, ymax),),     # keep y continuous; you could also break y similarly
            fig=fig,
            wspace=0.05, hspace=0.05
        )
        ax = bax  # use AX as the generic handle
    else:
        ax = fig.add_subplot(111)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        

    ax = fig.add_subplot(111)

    sc_base = ax.scatter(xs_base, ys_base, color=BASE, label="Base Models (no ft)")
    sc_ft = ax.scatter(xs_ft, ys_ft, color=FT, label="Fine-Tuned Models (ft)")


    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("Safe accuracy (h > 0)")
    ax.set_ylabel("Unsafe accuracy (h < 0)")

    title = f"Safe vs. Unsafe Accuracy — {env}"
    if title_suffix:
        title += f" ({title_suffix})"
    ax.set_title(title)
    ax.grid(True, linestyle=":", linewidth=0.8)

    if annotate:
        # Nudge labels a bit to reduce overlap
        for x, y, lab in zip(xs_base, ys_base, labels_base):
            ax.annotate(lab, (x, y), xytext=(0, 8), textcoords="offset points",
                        ha="center", va="bottom", fontsize=9, color="black")

        # Labels for ft models
        for x, y, lab in zip(xs_ft, ys_ft, labels_ft):
            ax.annotate(lab, (x, y), xytext=(0, 8), textcoords="offset points",
                        ha="center", va="bottom", fontsize=9, color="black")

    # === Legend for red and green categories ===
    red_patch = mpatches.Patch(color=FT, label="Fine-Tuned (ft)")
    green_patch = mpatches.Patch(color=BASE, label="Frozen (no ft)")
    ax.legend(handles=[green_patch, red_patch], loc="lower left")

    if tight_layout:
        fig.tight_layout()

    outpath = outdir / f"{env.replace(' ', '_')}.{fmt}"
    fig.savefig(outpath, dpi=200)
    plt.close(fig)
    return outpath

def make_intervention_rate_barplot(csv_path: Path,
                                   outdir: Path,
                                   fmt: str,
                                   env: str,
                                   title_suffix: Optional[str] = None,
                                   tight_layout: bool = True,
                                   annotate: bool = True,) -> Path:
    """
    Horizontal bar plot of avg_intervention_rate per backbone family for 'critic' rows.
    12 bars total: 6 families × {base (hollow), ft (filled)}, grouped by backbone.
    Y-ticks show the backbone family only; legend specifies base vs ft.
    """
    # --- Load CSV ---
    rows = []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            rows.append({
                "type": (row.get("type") or "").strip(),
                "finetune": _to_bool(row.get("finetune")),
                "backbone": (row.get("backbone") or "").strip(),
                "avg_intervention_rate": _to_float_or_none(row.get("avg_intervention_rate")),
            })

    # critic-only with numeric rate
    crit = [r for r in rows
            if r["type"] == "critic"
            and r["backbone"]
            and r["avg_intervention_rate"] is not None]
    if not crit:
        raise SystemExit("No valid 'critic' rows with avg_intervention_rate found.")

    # Canonical families (targeting 6 → 12 bars)
    canonical_families = ["dino", "dino_cls", "vc1", "r3m", "resnet", "scratch"]

    # Aggregate by family × (base/ft)
    from collections import defaultdict
    agg = defaultdict(lambda: {"base": [], "ft": []})
    for r in crit:
        fam = clean_backbone_name(r["backbone"])
        if fam in canonical_families:
            is_ft = r["finetune"] or r["backbone"].endswith("_ft")
            key = "ft" if is_ft else "base"
            agg[fam][key].append(r["avg_intervention_rate"])

    # Keep only families with both base and ft; cap at 6
    families = [fam for fam in canonical_families if agg[fam]["base"] and agg[fam]["ft"]][:6]
    if not families:
        raise SystemExit("No backbone families with BOTH base and ft for avg_intervention_rate (critic only).")

    # Prepare grouped data aligned by family
    base_vals = [mean(agg[f]["base"]) for f in families]
    ft_vals   = [mean(agg[f]["ft"])   for f in families]

    # --- Plot (grouped horizontal bars) ---
    fig, ax = plt.subplots(figsize=(9.5, 7.0))

    y_idx = np.arange(len(families))
    bar_h = 0.35
    offset = 0.22  # vertical offset for the pair split
    y_base = y_idx - offset
    y_ft   = y_idx + offset


    bars_base = ax.barh(y=y_base, width=base_vals, height=bar_h,
                        facecolor="none", edgecolor=BASE, linewidth=1.8)
    bars_ft   = ax.barh(y=y_ft,   width=ft_vals,   height=bar_h,
                        facecolor=FT, edgecolor="#444", linewidth=1.0)

    # Y ticks show each backbone family ONCE
    ax.set_yticks(y_idx)
    ax.set_yticklabels(families)

    # X limits with padding (rates in [0,1])
    xmax = max(base_vals + ft_vals) if (base_vals or ft_vals) else 1.0
    right_pad = 0.10 * max(1.0, xmax)
    ax.set_xlim(0, min(1.0, xmax + right_pad))

    # Grid, labels, title
    ax.grid(True, axis="x", linestyle=":", linewidth=0.8, alpha=0.8)
    ax.set_xlabel("Average Intervention Rate")
    title = "Average Intervention Rate by Backbone — " + env
    if title_suffix:
        title += f" ({title_suffix})"
    ax.set_title(title)

    # Optional value annotations at bar ends
    if annotate:
        def _annotate(bars, vals):
            for rect, v in zip(bars, vals):
                x = rect.get_width()
                y = rect.get_y() + rect.get_height()/2
                ax.text(min(x + 0.02, 0.98), y, f"{v:.3f}",
                        va="center", ha="left", fontsize=9)
        _annotate(bars_base, base_vals)
        _annotate(bars_ft,   ft_vals)

    # Legend: hollow vs filled
    leg_handles = [
        Line2D([0],[0], marker="s", linestyle="None",
               markerfacecolor="none", markeredgecolor=BASE,
               markeredgewidth=1.8, markersize=10, label="Base (no ft)"),
        Line2D([0],[0], marker="s", linestyle="None",
               markerfacecolor=FT, markeredgecolor="#444",
               markersize=10, label="Fine-Tuned (_ft)")
    ]
    fig.subplots_adjust(bottom=0.18, left=0.25, right=0.98, top=0.90)
    fig.legend(leg_handles, ["Base (no ft)", "Fine-Tuned (_ft)"],
               loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.04))

    if tight_layout:
        fig.tight_layout(rect=(0.25, 0.10, 0.98, 0.90))

    outpath = outdir / f"bar_intervention_rate_critic_grouped.{fmt}"
    fig.savefig(outpath, dpi=200)
    plt.close(fig)
    return outpath

def make_4panel_safety_scatter(rows: List[Dict[str, Any]],
                               outdir: Path,
                               fmt: str,
                               title_suffix: Optional[str],
                               tight_layout: bool) -> Path:
    marker_cycle = ['o','D','s','^','v','P','X','*','h','H','>','<']
    ms = 70  # marker size

    # Group data
    env_map = group_by_env(rows)
    envs = sorted(env_map.keys())

    # Marker per *cleaned* backbone (one shape per family)
    backbone_families = sorted({clean_backbone_name(r["backbone"]) for r in rows})
    marker_map = {bb: marker_cycle[i % len(marker_cycle)] for i, bb in enumerate(backbone_families)}

    # Figure + 2x2 axes (no shared axes → per-subplot ranges)
    fig, axes = plt.subplots(2, 2, figsize=(8, 6), sharex=False, sharey=False)
    axes = axes.flatten()
    for ax in axes:
        ax.set_visible(False)  # hide unused panes

    # Per-env padding
    pad_left_x, pad_right_x = 0.02, 0.12
    pad_bottom_y, pad_top_y = 0.02, 0.02

    for ax, env in zip(axes, envs):
        ax.set_visible(True)
        items = env_map[env]

        # Split plotting by tuning; assign shapes by backbone family
        for r in items:
            bb = clean_backbone_name(r["backbone"])
            mk = marker_map[bb]
            is_ft = r["backbone"].endswith("_ft")
            if is_ft:
                ax.scatter(r["safe_acc"], r["unsafe_acc"],
                           marker=mk, s=ms, facecolors=FT, edgecolors="#444",
                           linewidths=1.0)
            else:
                ax.scatter(r["safe_acc"], r["unsafe_acc"],
                           marker=mk, s=ms, facecolors="none", edgecolors=BASE,
                           linewidths=1.5)

        # ----- per-subplot limits with padding (clamped to [0,1])
        xs_env = [r["safe_acc"] for r in items]
        ys_env = [r["unsafe_acc"] for r in items]
        xmin, xmax = _lims_with_padding(xs_env, lo_pad_frac=0.06, hi_pad_frac=0.18,
                                        lo_clip=0.0, hi_clip=1.0)
        ymin, ymax = _lims_with_padding(ys_env, lo_pad_frac=0.06, hi_pad_frac=0.08,
                                        lo_clip=0.0, hi_clip=1.0)


        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

        # Diagonal reference clipped to current limits
        lo = max(xmin, ymin)
        hi = min(xmax, ymax)
        ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1, color="gray")

        ax.grid(True, linestyle=":", linewidth=0.8)
        ax.set_title(f"{env}" + (f" — {title_suffix}" if title_suffix else ""))

    # Axis labels (bottom row / left col)
    axes[2].set_xlabel("Safe accuracy (h > 0)")
    axes[3].set_xlabel("Safe accuracy (h > 0)")
    axes[0].set_ylabel("Unsafe accuracy (h < 0)")
    axes[2].set_ylabel("Unsafe accuracy (h < 0)")

    # Leave space for the legend under the grid
    fig.subplots_adjust(bottom=0.20, left=0.12, right=0.98, top=0.92)
    fig.subplots_adjust(left=0.14, right=0.98, top=0.92, bottom=0.22)
    if tight_layout:
        fig.tight_layout(rect=(0.14, 0.10, 0.98, 0.94))

    # ----- Single legend for the whole figure -----
    from matplotlib.lines import Line2D
    shape_handles = [
        Line2D([0], [0], marker=marker_map[bb], linestyle="None",
               markerfacecolor="none", markeredgecolor="#444", markersize=9, label=bb)
        for bb in backbone_families
    ]
    style_handles = [
        Line2D([0], [0], marker='o', linestyle="None",
               markerfacecolor="none", markeredgecolor=BASE,
               markersize=9, label="Base (no ft)"),
        Line2D([0], [0], marker='o', linestyle="None",
               markerfacecolor=FT, markeredgecolor="#444",
               markersize=9, label="Fine-Tuned (_ft)"),
    ]
    handles = shape_handles + style_handles
    fig.legend(handles=handles, loc="lower center",
               ncol=min(6, len(handles)), frameon=False, bbox_to_anchor=(0.5, 0.02))

    fig.suptitle("Safe vs. Unsafe Accuracy by Environment", y=0.995)
    if tight_layout:
        fig.tight_layout(rect=(0.12, 0.08, 0.98, 0.94))

    outpath = outdir / f"safe_unsafe_all_envs.{fmt}"
    fig.savefig(outpath, dpi=200)
    plt.close(fig)
    return outpath


def make_violation_success_4panel(rows: List[Dict[str, Any]],
                                  outdir: Path,
                                  fmt: str,
                                  title_suffix: Optional[str],
                                  tight_layout: bool) -> Path:
    # Keep rows that have both values (CARLA will be skipped)
    usable = [r for r in rows if (r.get("avg_violation") is not None and r.get("avg_success") is not None)]
    if not usable:
        raise SystemExit("No rows with avg_violation & avg_success found.")


    # Marker per backbone family
    families = sorted({clean_backbone_name(r["backbone"]) for r in usable})
    marker_cycle = ['o','D','s','^','v','P','X','*','h','H','>','<']
    marker_map = {bb: marker_cycle[i % len(marker_cycle)] for i, bb in enumerate(families)}
    ms = 70

    # Group by env
    env_map: Dict[str, List[Dict[str, Any]]] = {}
    for r in usable:
        env_map.setdefault(r["env"], []).append(r)
    envs = sorted(env_map.keys())

    # Figure (no shared axes → per-subplot ranges)
    fig, axes = plt.subplots(2, 2, figsize=(8, 6), sharex=False, sharey=False)
    axes = axes.flatten()
    for ax in axes:
        ax.set_visible(False)

    # Per-env padding (x can be large; y in [0,1])
    pad_left_x, pad_right_x = 0.03, 0.08
    pad_bottom_y, pad_top_y = 0.02, 0.06

    for ax, env in zip(axes, envs):
        ax.set_visible(True)
        items = env_map[env]

        for r in items:
            x = r["avg_violation"]
            y = r["avg_success"]
            bb = clean_backbone_name(r["backbone"])
            mk = marker_map[bb]
            is_ft = r["backbone"].endswith("_ft")

            if is_ft:
                ax.scatter(x, y, marker=mk, s=ms,
                           facecolors=FT, edgecolors="#444", linewidths=1.0)
            else:
                ax.scatter(x, y, marker=mk, s=ms,
                           facecolors="none", edgecolors=BASE, linewidths=1.5)

        # ----- per-subplot limits with padding
        xs_env = [r["avg_violation"] for r in items]
        ys_env = [r["avg_success"]   for r in items]
        xmin, xmax = _lims_with_padding(xs_env, lo_pad_frac=0.06, hi_pad_frac=0.15,
                                        lo_clip=0.0, hi_clip=None)
        ymin, ymax = _lims_with_padding(ys_env, lo_pad_frac=0.06, hi_pad_frac=0.08,
                                        lo_clip=0.0, hi_clip=1.0)
        

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

        ax.grid(True, linestyle=":", linewidth=0.8)
        ax.set_title(f"{env}" + (f" — {title_suffix}" if title_suffix else ""))

    # Labels
    axes[2].set_xlabel("Avg violation")
    axes[3].set_xlabel("Avg violation")
    axes[0].set_ylabel("Avg success")
    axes[2].set_ylabel("Avg success")

    # --- Single figure legend (shapes + fill/hollow) ---
    from matplotlib.lines import Line2D
    shape_handles = [
        Line2D([0], [0], marker=marker_map[bb], linestyle="None",
               markerfacecolor="none", markeredgecolor="#444",
               markersize=9, label=bb)
        for bb in families
    ]
    style_handles = [
        Line2D([0], [0], marker='o', linestyle="None",
               markerfacecolor="none", markeredgecolor=BASE,
               markersize=9, label="Base (no ft)"),
        Line2D([0], [0], marker='o', linestyle="None",
               markerfacecolor=FT, markeredgecolor="#444",
               markersize=9, label="Fine-Tuned (_ft)"),
    ]
    handles = shape_handles + style_handles

    fig.subplots_adjust(left=0.12, right=0.98, top=0.92, bottom=0.18)
    fig.legend(handles=handles, loc="lower center", ncol=min(6, len(handles)),
               frameon=False, bbox_to_anchor=(0.5, 0.04), prop={'size': 9})

    fig.suptitle("Avg Violation vs Avg Success by Environment", y=0.975)

    if tight_layout:
        fig.tight_layout(rect=(0.12, 0.10, 0.98, 0.93))

    outpath = outdir / f"violation_success_all_envs.{fmt}"
    fig.savefig(outpath, dpi=200)
    plt.close(fig)
    return outpath



def main():
    p = argparse.ArgumentParser(description="Plot Safe (x) vs Unsafe (y) accuracy per environment.")
    p.add_argument("--input", required=True, help="Path to CSV or JSON with fields: env, backbone, safe_acc, unsafe_acc")
    p.add_argument("--outdir", default="plots", help="Directory to save figures (default: plots)")
    p.add_argument("--format", dest="fmt", default="png", choices=["png", "pdf", "svg"], help="Output file format")
    p.add_argument("--annotate", action="store_true", help="Annotate points with backbone names")
    p.add_argument("--title-suffix", default=None, help="Optional suffix to append to figure titles (e.g., 'val set')")
    p.add_argument("--tight", action="store_true", help="Use tight_layout to reduce margins")
    p.add_argument("--label-format", choices=["backbone", "env", "env-backbone"],
               default="env-backbone", help="Annotation text")

    args = p.parse_args()

    inpath = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rows = read_input(inpath)
    validate_rows(rows)
    env_map = group_by_env(rows)

    if not env_map:
        raise SystemExit("No rows found. Ensure the input has at least one entry.")

    outputs = []
    for env, items in env_map.items():
        outpath = make_scatter_for_env(env, items, outdir, args.fmt, args.annotate, args.tight, args.title_suffix, args.label_format)
        outputs.append(str(outpath))

    print("Saved plots:")
    for pth in outputs:
        print(" -", pth)
    outpath = make_4panel_safety_scatter(rows, outdir, args.fmt, args.title_suffix, args.tight)
    print("Saved plot:")
    print(" -", outpath)

    outpath2 = make_violation_success_4panel(rows, outdir, args.fmt, args.title_suffix, args.tight)
    print("Saved plot:")
    print(" -", outpath2)

    outpath3 = make_intervention_rate_barplot(
        csv_path=Path("close_loop/csv/cargoal.csv"),
        outdir=outdir,
        fmt=args.fmt,
        env="Cargoal",
        title_suffix=args.title_suffix,
        tight_layout=args.tight
    )
    print("Saved plot:")
    print(" -", outpath3)

if __name__ == "__main__":
    main()
