"""Aggregate analyze_kv_drift.py records and produce per-task drift plots.

Layout per task: 2x2 grid
  rows: drift_from_warmup vs drift_lag1
  cols: K, V

In each subplot:
  x: decoding step
  y: drift (mean over samples)
  lines: position class × layer group (shallow=0-9, mid=10-21, deep=22-31)

Shaded bands: ±1 SE across samples.
"""

import argparse
import json
import os
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt


CLASSES = ["prompt", "gen_unmasked", "gen_masked"]
LAYER_GROUPS = {
    "shallow": list(range(0, 10)),
    "mid":     list(range(10, 22)),
    "deep":    list(range(22, 32)),
}
CLASS_COLOR = {"prompt": "tab:blue", "gen_unmasked": "tab:green", "gen_masked": "tab:orange"}
GROUP_STYLE = {"shallow": ":", "mid": "--", "deep": "-"}


def load_records(path):
    recs = []
    with open(path) as f:
        for line in f:
            recs.append(json.loads(line))
    return recs


def aggregate(recs, metric):
    """Returns dict[(class, group)] -> (steps_array, mean_array, sem_array).

    For each (sample, step, class, group): mean over layers in group.
    Then mean ± SEM over samples per step.
    """
    # Stage 1: per-sample per-step per-(class, group): mean over layers in group
    # bucket[(sample, step, class, group)] = list of metric values (one per layer in group)
    bucket = defaultdict(list)
    for r in recs:
        m = r.get(metric)
        if m is None or (isinstance(m, float) and (m != m)):  # NaN
            continue
        for gname, layers in LAYER_GROUPS.items():
            if r["layer"] in layers:
                bucket[(r["sample_id"], r["step"], r["class"], gname)].append(m)

    # Stage 2: per-(sample, step, class, group) -> single mean
    per_sample = defaultdict(dict)  # (class, group)[step] -> list of sample-level means
    for (sid, step, cls, gname), vals in bucket.items():
        per_sample[(cls, gname)].setdefault(step, []).append(float(np.mean(vals)))

    # Stage 3: across samples, mean ± SEM per step
    out = {}
    for key, by_step in per_sample.items():
        steps = sorted(by_step.keys())
        means = np.array([np.mean(by_step[s]) for s in steps])
        if len(next(iter(by_step.values()))) > 1:
            sems = np.array([np.std(by_step[s], ddof=1) / np.sqrt(len(by_step[s])) for s in steps])
        else:
            sems = np.zeros_like(means)
        out[key] = (np.array(steps), means, sems)
    return out


def plot_task(recs, task_label, save_path):
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    metrics = [
        ("drift_K_from_warmup", "K drift from warmup (1 - cos)"),
        ("drift_V_from_warmup", "V drift from warmup (1 - cos)"),
        ("drift_K_lag1",        "K lag-1 drift (1 - cos to prev step)"),
        ("drift_V_lag1",        "V lag-1 drift (1 - cos to prev step)"),
    ]
    for ax, (metric, title) in zip(axes.flat, metrics):
        agg = aggregate(recs, metric)
        for (cls, gname), (steps, means, sems) in sorted(agg.items()):
            ax.plot(steps, means,
                    color=CLASS_COLOR[cls],
                    linestyle=GROUP_STYLE[gname],
                    label=f"{cls} ({gname})", linewidth=1.6, alpha=0.95)
            ax.fill_between(steps, means - sems, means + sems,
                            color=CLASS_COLOR[cls], alpha=0.10)
        ax.set_title(title, fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("decoding step")
        ax.set_ylabel("drift (1 - cos)")
    # one legend
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"K/V drift during baseline full-forward decoding — {task_label}",
                 fontsize=12, y=1.00)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.13)
    fig.savefig(save_path, dpi=130, bbox_inches="tight")
    print(f"saved {save_path}")


def summary_table(recs):
    """Print a quick numeric summary: drift_from_warmup for prompt across deep
    layers at selected steps."""
    print("\n=== prompt deep drift_K_from_warmup ===")
    agg = aggregate(recs, "drift_K_from_warmup")
    if ("prompt", "deep") in agg:
        steps, m, s = agg[("prompt", "deep")]
        idxs = [0, 1, 5, 10, 20, 30, 40, 50, len(steps)-1]
        for i in idxs:
            if 0 <= i < len(steps):
                print(f"  step {int(steps[i]):3d}: {m[i]:.4f} ± {s[i]:.4f}")
    print("\n=== prompt deep drift_V_from_warmup ===")
    agg = aggregate(recs, "drift_V_from_warmup")
    if ("prompt", "deep") in agg:
        steps, m, s = agg[("prompt", "deep")]
        idxs = [0, 1, 5, 10, 20, 30, 40, 50, len(steps)-1]
        for i in idxs:
            if 0 <= i < len(steps):
                print(f"  step {int(steps[i]):3d}: {m[i]:.4f} ± {s[i]:.4f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="jsonl produced by analyze_kv_drift.py")
    p.add_argument("--label", required=True, help="task label for plot title")
    p.add_argument("--out", required=True, help="output png path")
    args = p.parse_args()

    recs = load_records(args.input)
    print(f"loaded {len(recs)} records")
    summary_table(recs)
    plot_task(recs, args.label, args.out)


if __name__ == "__main__":
    main()
