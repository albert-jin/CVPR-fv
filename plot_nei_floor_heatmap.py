"""Reproduce the NEI-floor/fusion-strength sensitivity heatmaps (Figure 6)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


GAMMAS = [0.0, 0.05, 0.1, 0.2, 0.3]
LAMBDAS = [0, 0.5, 1, 2, 4]

PANELS = {
    "SciFACT (32-shot Macro-F1)": np.array(
        [
            [65.8, 66.2, 66.7, 66.4, 65.9],
            [62.8, 66.5, 67.0, 67.2, 66.2],
            [63.4, 64.9, 66.8, 66.5, 66.0],
            [65.2, 65.5, 64.9, 63.2, 64.5],
            [64.6, 64.8, 63.6, 63.8, 63.4],
        ]
    ),
    "VitaminC (32-shot Macro-F1)": np.array(
        [
            [70.1, 71.9, 72.4, 72.1, 58.5],
            [70.5, 73.2, 72.7, 72.0, 60.8],
            [69.6, 72.0, 72.5, 72.3, 71.6],
            [71.2, 70.8, 71.2, 70.5, 69.8],
            [70.9, 69.5, 69.8, 68.9, 67.5],
        ]
    ),
    "FEVER (32-shot Macro-F1)": np.array(
        [
            [93.2, 90.1, 93.8, 93.4, 92.1],
            [90.4, 94.5, 95.2, 93.9, 92.5],
            [87.9, 93.9, 94.0, 93.6, 92.2],
            [89.3, 92.1, 92.5, 90.8, 90.5],
            [92.7, 89.0, 90.2, 89.4, 88.9],
        ]
    ),
}


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.15), constrained_layout=True)

    for ax, (title, values) in zip(axes, PANELS.items()):
        image = ax.imshow(values, cmap="viridis", aspect="auto")
        midpoint = (float(values.min()) + float(values.max())) / 2.0
        for row in range(values.shape[0]):
            for col in range(values.shape[1]):
                color = "white" if values[row, col] < midpoint else "black"
                ax.text(col, row, f"{values[row, col]:.1f}", ha="center", va="center", color=color)

        ax.set_title(title, pad=8)
        ax.set_xlabel(r"Fusion strength $\lambda$")
        ax.set_ylabel(r"NEI floor $\gamma$")
        ax.set_xticks(range(len(LAMBDAS)), [str(value) for value in LAMBDAS])
        ax.set_yticks(range(len(GAMMAS)), [str(value) for value in GAMMAS])
        colorbar = fig.colorbar(image, ax=ax, fraction=0.048, pad=0.035)
        colorbar.set_label("Macro-F1 (%)")

    output = Path(__file__).resolve().parents[3] / "nei_floor_lambda_heatmap-score.pdf"
    fig.savefig(output, bbox_inches="tight")
    print(output)


if __name__ == "__main__":
    main()
