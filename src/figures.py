"""Render publication figures from numbers.json (post + paper share them).

    python figures.py --numbers numbers.json --out figures/

Figures (each saved as PNG @ 200 dpi):
  fig1_saa_vs_accuracy.png     leaderboard scatter: accuracy vs SAA, CIs
  fig2_variant_accuracy.png    grouped bars: per-variant accuracy per model
  fig3_instability.png         run vs prompt instability per model
  fig4_flip_heatmap.png        item x model flip matrix
  fig5_position_bias.png       shuffled accuracy by gold-answer position
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

MODEL_COLORS = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3",
    "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD",
]


def _models(numbers: dict) -> list[dict]:
    return sorted(numbers["leaderboard"], key=lambda r: r["saa"], reverse=True)


def fig1_saa_vs_accuracy(numbers: dict, out: Path) -> None:
    rows = _models(numbers)
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    for i, r in enumerate(rows):
        x = r["accuracy"]
        y = r["saa"]
        ci = r.get("accuracy_ci") or [x, x]
        ax.errorbar(x, y, xerr=[[x - ci[0]], [ci[1] - x]],
                    fmt="o", ms=7, capsize=3,
                    color=MODEL_COLORS[i % len(MODEL_COLORS)],
                    label=r["model"])
    lim = 1.05
    ax.plot([0, lim], [0, lim], "--", color="gray", lw=1,
            label="SAA = accuracy (perfect stability)")
    ax.set_xlim(-0.02, lim)
    ax.set_ylim(-0.02, lim)
    ax.set_xlabel("Accuracy (fraction correct, with 95% cluster-bootstrap CI)")
    ax.set_ylabel("Stability-Adjusted Accuracy (SAA)")
    ax.set_title("Same answers, five phrasings: models sit below the diagonal\n"
                 "by exactly how much they flip")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "fig1_saa_vs_accuracy.png", dpi=200)
    plt.close(fig)


def fig2_variant_accuracy(numbers: dict, out: Path) -> None:
    rows = _models(numbers)
    variants = numbers["benchmark"]["variants"]
    x = np.arange(len(rows))
    width = 0.8 / len(variants)
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    for j, v in enumerate(variants):
        vals = [r["per_variant_accuracy"].get(v, 0.0) for r in rows]
        ax.bar(x + j * width - 0.4 + width / 2, vals, width, label=v)
    ax.set_xticks(x)
    ax.set_xticklabels([r["model"] for r in rows], rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("Accuracy by phrasing variant — the spread within a model\n"
                 "is its phrasing sensitivity")
    ax.legend(ncols=len(variants), fontsize=8, loc="lower left")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "fig2_variant_accuracy.png", dpi=200)
    plt.close(fig)


def fig3_instability(numbers: dict, out: Path) -> None:
    rows = _models(numbers)
    y = np.arange(len(rows))
    height = 0.38
    fig, ax = plt.subplots(figsize=(7.4, 0.55 * len(rows) + 1.8))
    ax.barh(y + height / 2, [r["prompt_instability"] for r in rows], height,
            label="prompt instability (across phrasings)", color="#C44E52")
    ax.barh(y - height / 2, [r["run_instability"] for r in rows], height,
            label="run instability (identical prompt, 2 repeats)", color="#4C72B0")
    ax.set_yticks(y)
    ax.set_yticklabels([r["model"] for r in rows], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Fraction of disagreeing pairs")
    ax.set_title("Where disagreement comes from: phrasing vs. the backend itself")
    ax.legend(fontsize=8)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "fig3_instability.png", dpi=200)
    plt.close(fig)

# %% PART2
def fig4_flip_heatmap(numbers: dict, out: Path) -> None:
    rows = _models(numbers)
    # Item order: by total flips across models (most-flippy on top).
    item_ids = sorted(
        {i for r in rows for i in r["flip_items"]},
        key=lambda it: -sum(1 for r in rows if it in r["flip_items"]),
    )
    if not item_ids:
        item_ids = []
    mat = np.array(
        [[1 if i in r["flip_items"] else 0 for i in item_ids] for r in rows]
    ) if item_ids else np.zeros((len(rows), 1))
    fig, ax = plt.subplots(figsize=(max(6.0, 0.35 * max(len(item_ids), 1) + 3),
                                    0.5 * len(rows) + 1.6))
    ax.imshow(mat, cmap="Reds", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(item_ids)))
    ax.set_xticklabels(item_ids, rotation=90, fontsize=7)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r["model"] for r in rows], fontsize=8)
    ax.set_title("Which items flip (answer changes across phrasings)")
    for ri in range(mat.shape[0]):
        for ci in range(mat.shape[1]):
            if mat[ri, ci]:
                ax.text(ci, ri, "X", ha="center", va="center",
                        color="white", fontsize=6)
    fig.tight_layout()
    fig.savefig(out / "fig4_flip_heatmap.png", dpi=200)
    plt.close(fig)


def fig5_position_bias(numbers: dict, out: Path) -> None:
    rows = [r for r in _models(numbers)
            if any(v is not None for v in r["shuffled_position_accuracy"].values())]
    if not rows:
        return
    positions = ["A", "B", "C", "D"]
    x = np.arange(len(positions))
    width = 0.8 / len(rows)
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for i, r in enumerate(rows):
        vals = [r["shuffled_position_accuracy"].get(p) or 0.0 for p in positions]
        ax.bar(x + i * width - 0.4 + width / 2, vals, width,
               label=r["model"], color=MODEL_COLORS[i % len(MODEL_COLORS)])
    ax.set_xticks(x)
    ax.set_xticklabels([f"gold at {p}" for p in positions])
    ax.set_ylabel("Accuracy (options-shuffled cells)")
    ax.set_ylim(0, 1.05)
    ax.set_title("Position bias: same question, gold answer moved")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "fig5_position_bias.png", dpi=200)
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--numbers", default="numbers.json")
    ap.add_argument("--out", default="figures")
    args = ap.parse_args(argv)
    numbers = json.loads(Path(args.numbers).read_text(encoding="utf-8"))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fig1_saa_vs_accuracy(numbers, out)
    fig2_variant_accuracy(numbers, out)
    fig3_instability(numbers, out)
    fig4_flip_heatmap(numbers, out)
    fig5_position_bias(numbers, out)
    for f in sorted(out.glob("fig*.png")):
        print(f"wrote {f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

