"""Spot with five holes: the input, the threshold (γ=0), the graph cut, the graph
cut with faired caps, and the original. Below, a close-up of the face.

The loose vertices that ``random_holes`` leaves are removed from the input: they
are points of the real cow and would give the method information that it does
not have.

    python demos/demo_segmentation_spot.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gsd import Mesh, connected_components, load_obj
from gsd.conforming import conforming_tet_mesh
from gsd.fairing import fair_filled_patches
from gsd.segmentation import graphcut_segment
from gsd.shapes import random_holes

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "figures"
MODEL = ROOT / "assets" / "spot" / "spot_triangulated.obj"

RED = np.array([0.65, 0.26, 0.18])
BLUE = np.array([0.16, 0.35, 0.55])
GREEN = np.array([0.20, 0.47, 0.27])
VIOLET = np.array([0.42, 0.30, 0.58])
GRAY = np.array([0.55, 0.58, 0.60])

SIZE = 1.35


def load_spot() -> Mesh:
    m = load_obj(MODEL)
    lo, hi = m.bounding_box()
    scale = SIZE / float((hi - lo).max())
    V = (m.V - (lo + hi) / 2) * scale
    # The model has its vertical axis along y; it is permuted so that the cow stands.
    return Mesh(np.column_stack([V[:, 0], -V[:, 2], V[:, 1]]), m.F)


def draw(ax, V, F, color, title: str, limits):
    if len(F) == 0:
        ax.set_title(f"{title}\n(empty)", fontsize=10)
        ax.set_axis_off()
        return

    tris = V[F]
    a, b, c = tris[:, 0], tris[:, 1], tris[:, 2]
    n = np.cross(b - a, c - a)
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-300)

    light = np.array([0.5, 0.35, 0.79])
    light /= np.linalg.norm(light)
    intensity = 0.30 + 0.70 * np.clip(n @ light, 0.0, 1.0)

    ax.add_collection3d(Poly3DCollection(
        tris, facecolors=np.clip(color[None, :] * intensity[:, None], 0, 1),
        edgecolors="none",
    ))

    lo, hi = limits
    center = 0.5 * (lo + hi)
    radius = 0.62 * float((hi - lo).max())
    for k, ax_name in enumerate("xyz"):
        getattr(ax, f"set_{ax_name}lim")(center[k] - radius, center[k] + radius)

    ax.set_box_aspect((1, 1, 1), zoom=1.35)
    ax.view_init(elev=14, azim=-58)
    ax.set_proj_type("persp", focal_length=0.7)
    ax.set_axis_off()
    ax.set_title(title, fontsize=10)


def face_closeup(V, F, color) -> np.ndarray | None:
    """The head up close, rendered with pyvista (up close, matplotlib sorts the
    faces incorrectly). Without pyvista it returns None."""
    try:
        import pyvista as pv
    except ImportError:
        return None
    pv.OFF_SCREEN = True
    pv_mesh = pv.PolyData(V, np.hstack([np.full((len(F), 1), 3), F]).ravel())
    p = pv.Plotter(off_screen=True, window_size=(700, 640))
    p.set_background("white")
    p.add_mesh(pv_mesh, color=color, specular=0.25)
    # The head faces +y; it is seen in three-quarter view, from the side of the muzzle.
    p.camera_position = [(1.2, 0.9, 0.45), (0.0, 0.3, 0.3), (0.0, 0.0, 1.0)]
    image = p.screenshot(return_img=True)
    p.close()
    return image


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    spot = load_spot()
    limits = spot.bounding_box()

    broken = random_holes(spot, target_fraction=0.22, n_holes=5, seed=11)
    used, F_compact = np.unique(broken.F, return_inverse=True)
    broken = Mesh(broken.V[used], F_compact.reshape(-1, 3))
    missing = 1.0 - broken.total_area / spot.total_area
    print(f"Spot: {spot.n_faces} faces · broken: {broken.n_faces} faces "
          f"({missing:.0%} of the area missing in 5 holes)")

    t0 = time.perf_counter()
    mesh, _ = conforming_tet_mesh(broken)
    print(f"CDT: {mesh.n_tets} tets in {time.perf_counter()-t0:.1f}s")

    results = {}
    for name, gamma in (("threshold", 0.0), ("graphcut", 40.0)):
        t0 = time.perf_counter()
        res = graphcut_segment(broken, mesh=mesh, gamma=gamma)
        elapsed = time.perf_counter() - t0

        recon = Mesh(res["V"], res["F"])
        d = cKDTree(res["V"]).query(spot.V)[0]
        rel_area = recon.total_area / spot.total_area

        results[name] = res
        print(f"  {name:9s} (γ={gamma:g}): {len(res['F'])} faces, "
              f"{connected_components(recon)} components, "
              f"area {rel_area:.3f}×, fidelity {d.mean():.5f}, {elapsed:.1f}s")

    t0 = time.perf_counter()
    V_fair, F_fair = fair_filled_patches(results["graphcut"]["V"],
                                         results["graphcut"]["F"], broken)
    d = cKDTree(V_fair).query(spot.V)[0]
    print(f"  cap fairing: {len(F_fair)} faces, "
          f"area {Mesh(V_fair, F_fair).total_area / spot.total_area:.3f}×, "
          f"fidelity {d.mean():.5f}, {time.perf_counter()-t0:.1f}s")

    columns = [
        (broken.V, broken.F, RED, f"input — {missing:.0%} missing in 5 holes"),
        (results["threshold"]["V"], results["threshold"]["F"], BLUE,
         "plain threshold (γ=0)\nEquation 7"),
        (results["graphcut"]["V"], results["graphcut"]["F"], GREEN,
         "graph cut (γ=40)\nEquation 8"),
        (V_fair, F_fair, VIOLET, "graph cut + faired caps\nthe input does not move"),
        (spot.V, spot.F, GRAY, "original Spot\n(reference)"),
    ]
    closeups = [face_closeup(V, F, color) for V, F, color, _ in columns]
    rows = 2 if all(c is not None for c in closeups) else 1

    fig = plt.figure(figsize=(21.0, 4.6 * rows))
    width = 1.0 / len(columns)
    height = 1.0 / rows
    for k, (V, F, color, title) in enumerate(columns):
        draw(fig.add_axes([k * width, 1.0 - height, width, height * 0.95], projection="3d"),
             V, F, color, title, limits)
        if rows == 2:
            ax = fig.add_axes([k * width + 0.01, 0.0, width - 0.02, height * 0.92])
            ax.imshow(closeups[k])
            ax.set_axis_off()
            ax.set_title("the face, up close", fontsize=10)
    fig.suptitle("SPOT — robust segmentation by graph cut (Jacobson, "
                 "Kavan and Sorkine-Hornung 2013)", fontsize=12.5, y=1.04,
                 fontweight="bold")

    dest = OUTPUT / "segmentation_spot.png"
    fig.savefig(dest, dpi=145, bbox_inches="tight")
    plt.close(fig)
    print(f"\nfigure at {dest.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
