"""w along an axis through two spheres: with both complete, with a hole in the
second one, and with the second one deleted.

A hole degrades w near the piece; deleting the piece removes it from the sum, and
its contribution vanishes everywhere.

    python demos/demo_missing_piece.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gsd import icosphere, merge_meshes, punch_hole, winding_number
from gsd.mesh import Mesh

OUTPUT = Path(__file__).resolve().parent.parent / "figures"

BLACK = "#14161d"
RED = "#A6412F"
BLUE = "#2F6FB5"


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)

    a = icosphere(2)
    b = icosphere(2)
    b = Mesh(b.V + np.array([4.0, 0.0, 0.0]), b.F)

    xs = np.linspace(-1.3, 5.3, 300)
    pts = np.column_stack([xs, np.zeros_like(xs), np.zeros_like(xs)])

    complete = merge_meshes([a, b])
    w_complete = winding_number(complete, pts)

    b_holed = punch_hole(b, center=np.array([4.0, 0.0, 1.0]), radius=1.3)
    with_hole = merge_meshes([a, b_holed])
    w_hole = winding_number(with_hole, pts)

    w_deleted = winding_number(a, pts)

    far = xs < 1.5
    diff_far = np.abs(w_complete[far] - w_deleted[far]).max()
    i4 = np.argmin(np.abs(xs - 4.0))
    print(f"w at x=4: complete={w_complete[i4]:.4f}  "
          f"with hole={w_hole[i4]:.4f}  deleted={w_deleted[i4]:.6f}")
    print(f"largest difference near A (x<1.5) when deleting B: {diff_far:.2e}")

    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    ax.plot(xs, w_complete, color=BLACK, lw=2.0, label="both pieces complete")
    ax.plot(xs, w_hole, color=BLUE, lw=2.0, ls="--",
            label="B with a hole (still present)")
    ax.plot(xs, w_deleted, color=RED, lw=2.2,
            label="B deleted entirely")

    ax.axhline(0, color="black", lw=0.6, alpha=0.4)
    ax.axhline(1, color="black", lw=0.6, alpha=0.4)
    ax.axvline(0, color="black", lw=0.6, alpha=0.25)
    ax.axvline(4, color="black", lw=0.6, alpha=0.25)
    ax.annotate("center of A", (0, 1.06), ha="center", fontsize=9, color="black")
    ax.annotate("center of B", (4, 1.06), ha="center", fontsize=9, color="black")

    ax.set_xlabel("position along the axis")
    ax.set_ylabel("$w(x)$")
    ax.set_title("The contribution of a piece to $w$ elsewhere does not depend on "
                 "that piece\n(deleting it removes it from the sum; a hole keeps it, "
                 "degraded, in the sum)", fontsize=11)
    ax.legend(loc="center left", fontsize=9, framealpha=0.95)
    fig.tight_layout()

    dest = OUTPUT / "missing_piece.png"
    fig.savefig(dest, dpi=145)
    plt.close(fig)
    print(f"\nfigure at {dest.relative_to(OUTPUT.parent)}")


if __name__ == "__main__":
    main()
