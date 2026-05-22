"""Plot prompt-side drift binned by distance from gen boundary.

For each task: 2x2 grid (K vs V) × (from_warmup vs lag1).
Lines: distance bin × layer group.
"""

import argparse
import json
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt


BINS = ["prompt_d0_32", "prompt_d32_128", "prompt_d128_512", "prompt_d512_plus"]
BIN_LABELS = {
    "prompt_d0_32":     "d∈[0,32) (next to gen)",
    "prompt_d32_128":   "d∈[32,128)",
    "prompt_d128_512":  "d∈[128,512)",
    "prompt_d512_plus": "d≥512 (far)",
}
BIN_COLOR = {
    "prompt_d0_32":     "tab:red",
    "prompt_d32_128":   "tab:orange",
    "prompt_d128_512":  "tab:olive",
    "prompt_d512_plus": "tab:blue",
}
LAYER_GROUPS = {
    "shallow": list(range(0, 10)),
    "mid":     list(range(10, 22)),
    "deep":    list(range(22, 32)),
}
GROUP_STYLE = {"shallow": ":", "mid": "--", "deep": "-"}


def load(path):
    with open(path) as f:
        return [json.loads(l) for l in f]


def aggregate(recs, metric, cls):
    bucket = defaultdict(list)
    for r in recs:
        if r["class"] != cls: continue
        m = r.get(metric)
        if m is None: continue
        try:
            if np.isnan(m): continue
        except Exception:
            pass
        for gname, layers in LAYER_GROUPS.items():
            if r["layer"] in layers:
                bucket[(r["sample_id"], r["step"], gname)].append(m)

    per_step = defaultdict(dict)
    for (sid, step, g), vals in bucket.items():
        per_step[g].setdefault(step, []).append(float(np.mean(vals)))

    out = {}
    for g, by_step in per_step.items():
        steps = sorted(by_step)
        means = np.array([np.mean(by_step[s]) for s in steps])
        if len(next(iter(by_step.values()))) > 1:
            sems = np.array([np.std(by_step[s], ddof=1) / np.sqrt(len(by_step[s])) for s in steps])
        else:
            sems = np.zeros_like(means)
        out[g] = (np.array(steps), means, sems)
    return out


def plot(recs, task_label, save_path):
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    metrics = [
        ("drift_K_from_warmup", "K drift from warmup"),
        ("drift_V_from_warmup", "V drift from warmup"),
        ("drift_K_lag1",        "K lag-1 drift"),
        ("drift_V_lag1",        "V lag-1 drift"),
    ]
    for ax, (metric, title) in zip(axes.flat, metrics):
        # focus on deep layers only (cleanest signal for prompt)
        for bin_name in BINS:
            agg = aggregate(recs, metric, bin_name)
            # Only plot 'deep' to declutter (deep is where drift shows up)
            if "deep" not in agg:
                continue
            steps, means, sems = agg["deep"]
            ax.plot(steps, means, color=BIN_COLOR[bin_name],
                    linestyle="-", linewidth=1.8, alpha=0.95,
                    label=BIN_LABELS[bin_name])
            ax.fill_between(steps, means - sems, means + sems,
                            color=BIN_COLOR[bin_name], alpha=0.12)
        ax.set_title(title + " — DEEP layers (22-31)", fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("decoding step")
        ax.set_ylabel("drift (1 - cos)")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=9,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"Prompt-side K/V drift by distance from gen boundary — {task_label}",
                 fontsize=12, y=1.00)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.10)
    fig.savefig(save_path, dpi=130, bbox_inches="tight")
    print(f"saved {save_path}")


def summary(recs):
    print("\n=== Summary: deep K drift_from_warmup by distance bin (at selected steps) ===")
    print(f"{'step':>5}", end="")
    for b in BINS:
        print(f"  {b[7:]:>14}", end="")
    print()
    aggs = {b: aggregate(recs, "drift_K_from_warmup", b) for b in BINS}
    # Find common step axis from prompt_d0_32 deep
    if "deep" in aggs[BINS[0]]:
        steps0 = aggs[BINS[0]]["deep"][0]
        for show_step in [1, 5, 10, 20, 30, 50, 80, 99]:
            idx = (np.abs(steps0 - show_step)).argmin()
            if abs(int(steps0[idx]) - show_step) > 3:
                continue
            print(f"{int(steps0[idx]):>5}", end="")
            for b in BINS:
                if "deep" in aggs[b] and idx < len(aggs[b]["deep"][1]):
                    m = aggs[b]["deep"][1][idx]
                    print(f"  {m:>14.4f}", end="")
                else:
                    print(f"  {'-':>14}", end="")
            print()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    recs = load(args.input)
    print(f"loaded {len(recs)} records")
    summary(recs)
    plot(recs, args.label, args.out)


if __name__ == "__main__":
    main()
