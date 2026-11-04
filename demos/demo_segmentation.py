"""Graph cut against a plain threshold on a synthetic w: a smooth transition in x
plus independent noise in each tetrahedron. A slice colored by hit or miss is
shown.

    python demos/demo_segmentation.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gsd.segmentation import (
    data_term,
    estimate_sigma,
    min_cut_labels,
    smoothness_weights,
    tet_adjacency,
)
from gsd.tetmesh import tet_grid

OUTPUT = Path(__file__).resolve().parent.parent / "figures"

GREEN = "#2E7D46"
RED = "#A6412F"
GRAY = "#8A8A8A"


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)

    mesh = tet_grid(n=10, size=2.0)
    tet_a, tet_b, area = tet_adjacency(mesh)
    centers = mesh.tet_centroids

    w_true = 0.5 * (1 - np.tanh(centers[:, 0] / 0.35))
    rng = np.random.default_rng(3)
    w_noisy = np.clip(w_true + rng.normal(0, 0.22, size=len(w_true)), -0.4, 1.4)
    truth = centers[:, 0] < 0

    u_inside, u_outside = data_term(w_noisy)
    sigma = estimate_sigma(w_noisy, tet_a, tet_b)
    print(f"estimated sigma (85th percentile of |Δw| between neighbors): {sigma:.4f}")

    gammas = (0.0, 5.0, 40.0)
    labelings, accuracies = [], []
    for gamma in gammas:
        v = gamma * smoothness_weights(w_noisy, tet_a, tet_b, area, sigma)
        lab = min_cut_labels(u_inside, u_outside, tet_a, tet_b, v)
        labelings.append(lab)
        accuracies.append(float(np.mean(lab == truth)))
        print(f"  gamma={gamma:6.1f}  accuracy={accuracies[-1]:.4f}  "
              f"errors={(lab != truth).sum()}/{len(lab)}")

    in_slice = np.abs(centers[:, 2]) < 0.11
    x, y = centers[in_slice, 0], centers[in_slice, 1]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))

    sc = axes[0].scatter(x, y, c=w_noisy[in_slice], cmap="RdBu_r", vmin=0, vmax=1, s=14)
    axes[0].set_title("$w$ with incoherent noise\n(input)")
    fig.colorbar(sc, ax=axes[0], fraction=0.046)

    for ax, gamma, lab, acc in zip(axes[1:], gammas, labelings, accuracies):
        correct = lab[in_slice] == truth[in_slice]
        ax.scatter(x[correct], y[correct], c=GREEN, s=14, label="correct")
        ax.scatter(x[~correct], y[~correct], c=RED, s=22, marker="x",
                   label="error")
        name = "threshold (γ=0)" if gamma == 0 else f"graph cut (γ={gamma:g})"
        ax.set_title(f"{name}\naccuracy {acc:.1%}")
        ax.legend(loc="upper right", fontsize=8, framealpha=0.9)

    for ax in axes:
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle("Graph cut segmentation against a plain threshold, on "
                 "noise that is incoherent between neighbors", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUTPUT / "segmentation_graphcut.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\nfigure at {OUTPUT.relative_to(OUTPUT.parent)}/segmentation_graphcut.png")


if __name__ == "__main__":
    main()
