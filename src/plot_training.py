"""
Generate training curve figures from a TRL GRPO training log.

Usage:
    python -m src.plot_training --log training_3b_run.log --out figures/
"""

import argparse
import ast
import os
import re
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# --------------------------------------------------------------------------- #
# Parsing                                                                       #
# --------------------------------------------------------------------------- #

def parse_log(path: str) -> list[dict]:
    """Extract per-step metric dicts from a TRL training log."""
    records = []
    pattern = re.compile(r"\{[^{}]+\}")
    with open(path) as f:
        text = f.read()
    for m in pattern.finditer(text):
        chunk = m.group(0)
        try:
            # TRL logs single-quoted dicts; convert to proper JSON
            chunk_json = chunk.replace("'", '"')
            d = json.loads(chunk_json)
        except Exception:
            try:
                d = ast.literal_eval(chunk)
            except Exception:
                continue
        # Only keep step-level metric dicts (have 'reward' key)
        if "reward" in d and "loss" in d:
            skip = {"epoch"} | {k for k, v in d.items() if not isinstance(v, (int, float, str))}
            records.append({k: float(v) for k, v in d.items() if k not in skip})
    return records


# --------------------------------------------------------------------------- #
# Plotting                                                                      #
# --------------------------------------------------------------------------- #

PANEL_CFG = [
    ("reward",     "Composite Reward",       "royalblue",    (0.0, 1.0),   True),
    ("loss",       "Training Loss",           "tomato",       None,         False),
    ("kl",         "KL Divergence (β=0.04)", "mediumpurple", None,         False),
    ("entropy",    "Policy Entropy",          "seagreen",     (0.0, None),  False),
    ("grad_norm",  "Gradient Norm",           "darkorange",   None,         False),
    ("learning_rate", "Learning Rate",        "slategray",    None,         False),
]


def moving_avg(vals, w=3):
    out = []
    for i, v in enumerate(vals):
        window = vals[max(0, i - w + 1): i + 1]
        out.append(sum(window) / len(window))
    return out


def plot_training_curves(records: list[dict], out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    steps = list(range(1, len(records) + 1))

    # ── 1. Dashboard figure (6 panels) ───────────────────────────────────── #
    fig = plt.figure(figsize=(15, 10))
    fig.suptitle(
        "GRPO Training — Qwen2.5-3B-Instruct on τ²-bench Retail (74 tasks, CPU)\n"
        "1 epoch · lr=5e-6 · K=2 · LoRA r=16",
        fontsize=13, fontweight="bold", y=1.01,
    )
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    for idx, (key, title, color, ylim, draw_target) in enumerate(PANEL_CFG):
        ax = fig.add_subplot(gs[idx // 3, idx % 3])
        vals = [r.get(key, float("nan")) for r in records]

        ax.plot(steps, vals, color=color, alpha=0.35, linewidth=1.2)
        smoothed = moving_avg(vals, w=5)
        ax.plot(steps, smoothed, color=color, linewidth=2.2)

        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Training Step", fontsize=9)
        ax.grid(True, alpha=0.3)
        if ylim:
            lo, hi = ylim
            if lo is not None:
                ax.set_ylim(bottom=lo)
            if hi is not None:
                ax.set_ylim(top=hi)

    path_dashboard = os.path.join(out_dir, "training_curves_dashboard.png")
    fig.savefig(path_dashboard, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path_dashboard}")

    # ── 2. Reward curve standalone ────────────────────────────────────────── #
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    rewards = [r.get("reward", float("nan")) for r in records]
    ax2.plot(steps, rewards, color="royalblue", alpha=0.4, linewidth=1.2, label="per-step reward")
    ax2.plot(steps, moving_avg(rewards, w=5), color="royalblue", linewidth=2.5, label="smoothed (w=5)")
    ax2.axhline(0.7, color="red", linestyle="--", linewidth=1.5, label="target ≥ 0.7")
    ax2.fill_between(steps, moving_avg(rewards, w=5), 0.7,
                     where=[v >= 0.7 for v in moving_avg(rewards, w=5)],
                     alpha=0.15, color="green", label="above target")
    ax2.set_title("Composite Reward During GRPO Training\n"
                  "Qwen2.5-3B-Instruct · τ²-bench Retail Train Split",
                  fontsize=12, fontweight="bold")
    ax2.set_xlabel("Training Step", fontsize=11)
    ax2.set_ylabel("Composite Reward", fontsize=11)
    ax2.set_ylim(0.0, 1.0)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    path_reward = os.path.join(out_dir, "reward_curve.png")
    fig2.savefig(path_reward, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved: {path_reward}")

    # ── 3. Print summary stats ────────────────────────────────────────────── #
    print("\n=== Training Summary ===")
    print(f"  Steps completed : {len(records)}")
    for key, title, *_ in PANEL_CFG:
        vals = [r[key] for r in records if key in r]
        if vals:
            print(f"  {title:30s}  first={vals[0]:.4f}  last={vals[-1]:.4f}"
                  f"  mean={sum(vals)/len(vals):.4f}  max={max(vals):.4f}")


# --------------------------------------------------------------------------- #
# CLI                                                                           #
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", default="training_3b_run.log")
    parser.add_argument("--out", default="figures")
    args = parser.parse_args()

    records = parse_log(args.log)
    if not records:
        print("No step records found in log.")
        return
    print(f"Parsed {len(records)} training steps.")
    plot_training_curves(records, args.out)


if __name__ == "__main__":
    main()
