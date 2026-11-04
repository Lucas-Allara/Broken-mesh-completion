"""w on the plane y = 0 for an intact sphere and for the same sphere without a cap,
and its profile along the z axis.

    python demos/demo_winding.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gsd import icosphere, punch_hole, winding_number
from gsd.winding import sample_grid

OUTPUT = Path(__file__).resolve().parent.parent / "figures"

BLUE = "#28598C"
RED = "#A6412F"
HOLE_RADIUS = 0.6


def slice_plot(ax, mesh, title: str, res: int = 220):
    w, extent = sample_grid(mesh, resolution=res, padding=0.2, axis=1, offset=0.0)

    im = ax.imshow(
        w,
        extent=extent,
        origin="lower",
        cmap="RdBu_r",
        vmin=0.0,
        vmax=1.0,
        interpolation="bilinear",
    )
    ax.contour(w, levels=[0.5], extent=extent, colors="black", linewidths=1.2)

    ax.set_title(title, fontsize=11)
    ax.set_xlabel("x coordinate")
    ax.set_ylabel("z coordinate")
    ax.set_aspect("equal")
    return im


def profile_plot(ax, intact, broken):
    t = np.linspace(-1.6, 1.6, 600)
    P = np.stack([np.zeros_like(t), np.zeros_like(t), t], axis=1)

    ax.plot(t, winding_number(intact, P), color=BLUE, lw=1.8, label="closed sphere (intact)")
    ax.plot(t, winding_number(broken, P), color=RED, lw=1.8, label="with a missing cap")

    ax.axhline(0.5, color="black", lw=0.8, ls="--", alpha=0.5)
    ax.text(-1.55, 0.54, "w = 0.5 (inside/outside threshold)", fontsize=8, color="black", alpha=0.7)
    ax.axvline(1.0, color="gray", lw=0.8, ls=":", alpha=0.7)
    ax.text(1.02, -0.28, "north pole\n(where the hole is)", fontsize=8, color="gray")

    ax.set_title("profile of w along the z axis", fontsize=11)
    ax.set_xlabel("z coordinate")
    ax.set_ylabel("winding number w")
    ax.set_ylim(-0.4, 1.4)
    ax.legend(fontsize=9, loc="lower left")
    ax.grid(alpha=0.25)


def main() -> None:
    intact = icosphere(4)
    broken = punch_hole(intact, np.array([0.0, 0.0, 1.0]), HOLE_RADIUS)

    missing = 1.0 - broken.total_area / intact.total_area
    w_center = winding_number(broken, np.zeros((1, 3)))[0]

    print(f"faces: {intact.n_faces} intact, {broken.n_faces} broken")
    print(f"missing area: {missing:.4f}")
    print(f"winding number at the center of the broken sphere: {w_center:.4f}")
    print(f"  (predicted 1 - missing area = {1.0 - missing:.4f})")

    fig = plt.figure(figsize=(15, 4.4), constrained_layout=True)
    gs = fig.add_gridspec(1, 4, width_ratios=[1.0, 1.0, 0.05, 1.15])
    axes = [fig.add_subplot(gs[0, i]) for i in (0, 1)]
    cax = fig.add_subplot(gs[0, 2])

    im = slice_plot(axes[0], intact, "closed sphere")
    slice_plot(axes[1], broken, f"with a missing cap (r = {HOLE_RADIUS})")
    profile_plot(fig.add_subplot(gs[0, 3]), intact, broken)

    fig.colorbar(im, cax=cax)
    cax.set_title("w\n(0 to 1)", fontsize=9, pad=8)

    OUTPUT.mkdir(exist_ok=True)
    dest = OUTPUT / "stage1_winding.png"
    fig.savefig(dest, dpi=150, bbox_inches="tight")
    print(f"\nfigure written to {dest.relative_to(OUTPUT.parent)}")


if __name__ == "__main__":
    main()
