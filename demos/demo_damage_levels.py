"""Error in the holes for the method, the method with fairing and the method with
symmetry, on Spot and Homer with 5%, 10% and 22% of the surface missing (three
sets of holes per level).

On the same output of the method, it also measures fairing with the Laplacian
(as TetWild does), with Δ² and with three subdivisions instead of two.

Writes to ``figures/``: ``error_vs_damage.png``, ``damage_levels_spot.png``,
``damage_levels_homer.png`` and the numbers in ``damage_levels.json``.

    python demos/demo_damage_levels.py            # about 10 minutes
    python demos/demo_damage_levels.py --figures  # only redraws the saved results
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gsd.fairing import fair_filled_patches
from gsd.mesh import Mesh, load_obj
from gsd.shapes import random_holes
from gsd.symmetry import _samples, reconstruct_with_symmetry

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "figures"
DATA = OUTPUT / "level_data"

MODELS = {
    "spot": (ROOT / "assets" / "spot" / "spot_triangulated.obj", True),
    "homer": (ROOT / "assets" / "homer.obj", False),
}
LEVELS = (0.05, 0.10, 0.22)
SEEDS = (3, 7, 21)
FIGURE_SEED = 3
SIZE = 1.35

VARIANTS = ("method", "faired", "symmetry")
ALTERNATIVES = {"laplacian": {"order": 1}, "bilaplacian": {"order": 2}, "three_subdivisions": {"refine": 3}}
LABELS = {"method": "method of the paper", "faired": "+ fairing",
          "symmetry": "+ symmetry"}
COLORS = {"method": "#2a78d6", "faired": "#eb6834", "symmetry": "#1baf7a"}
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID_COLOR = "#e4e3df"


def load_model(name: str) -> Mesh:
    obj_path, permute = MODELS[name]
    m = load_obj(obj_path)
    lo, hi = m.bounding_box()
    V = (m.V - (lo + hi) / 2) * (SIZE / float((hi - lo).max()))
    if permute:  # Spot has its vertical axis along y
        V = np.column_stack([V[:, 0], -V[:, 2], V[:, 1]])
    return Mesh(V, m.F)


def punch_holes(truth: Mesh, fraction: float, hole_seed: int) -> tuple[Mesh, np.ndarray]:
    """Input with holes, without loose vertices, and the mask of the missing
    faces."""
    broken = random_holes(truth, target_fraction=fraction, n_holes=5, seed=hole_seed)
    used, F = np.unique(broken.F, return_inverse=True)
    broken = Mesh(broken.V[used], F.reshape(-1, 3))
    present = {tuple(sorted(map(tuple, p.tolist()))) for p in broken.V[broken.F]}
    missing = np.array([tuple(sorted(map(tuple, truth.V[t].tolist()))) not in present
                        for t in truth.F])
    return broken, missing


def hole_error(missing_truth: np.ndarray, V: np.ndarray, F: np.ndarray,
               rng: np.random.Generator) -> float:
    """Mean distance from the missing true surface to the reconstruction, as a
    percentage of the size of the model."""
    tree = cKDTree(_samples(V, F, 300_000, rng))
    return 100.0 * float(tree.query(missing_truth)[0].mean()) / SIZE


def run() -> list[dict]:
    DATA.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in MODELS:
        truth = load_model(name)
        for fraction in LEVELS:
            for hole_seed in SEEDS:
                rng = np.random.default_rng(0)
                broken, missing = punch_holes(truth, fraction, hole_seed)
                missing_pts = _samples(truth.V, truth.F[missing], 100_000, rng)
                t0 = time.perf_counter()
                out = reconstruct_with_symmetry(broken)
                seconds = time.perf_counter() - t0
                row = {"model": name, "level": fraction, "seed": hole_seed,
                       "missing": 1.0 - broken.total_area / truth.total_area,
                       "seconds": seconds}
                for v in VARIANTS:
                    VF = out[v] if out[v] is not None else out["faired"]
                    row[v] = hole_error(missing_pts, VF[0], VF[1], rng)
                for key, options in ALTERNATIVES.items():
                    VF = fair_filled_patches(*out["method"], broken, **options)
                    row[key] = hole_error(missing_pts, VF[0], VF[1], rng)
                row["mirrored"] = 0 if out["mirrored"] is None else out["mirrored"].n_faces
                rows.append(row)
                print(f"{name} {fraction:.0%} seed {hole_seed:2d}: "
                      + "  ".join(f"{v} {row[v]:.3f}%" for v in VARIANTS)
                      + f"  ({seconds:.0f}s)", flush=True)
                if hole_seed == FIGURE_SEED:
                    np.savez(DATA / f"{name}_{int(fraction * 100)}.npz",
                             input_V=broken.V, input_F=broken.F,
                             **{f"{v}_V": (out[v] or out["faired"])[0] for v in VARIANTS},
                             **{f"{v}_F": (out[v] or out["faired"])[1] for v in VARIANTS})
        np.savez(DATA / f"{name}_truth.npz", V=truth.V, F=truth.F)
    (OUTPUT / "damage_levels.json").write_text(json.dumps(rows, indent=1))
    return rows


def plot_errors(rows: list[dict]) -> None:
    """Error against the missing percentage: one panel per model, on the same scale."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 14, "axes.edgecolor": GRID_COLOR,
                         "axes.labelcolor": INK_SECONDARY,
                         "xtick.color": INK_SECONDARY, "ytick.color": INK_SECONDARY})
    fig, axes = plt.subplots(1, len(MODELS), figsize=(12, 4.8), sharey=True,
                             constrained_layout=True)
    x = 100 * np.array(LEVELS)
    for ax, name in zip(axes, MODELS):
        for k, v in enumerate(VARIANTS):
            values = np.array([[f[v] for f in rows if f["model"] == name and f["level"] == n]
                               for n in LEVELS])
            mean_err = values.mean(axis=1)
            # Range across the sets of holes, shifted so that they do not overlap.
            ax.vlines(x + 0.45 * (k - 1), values.min(axis=1), values.max(axis=1),
                      color=COLORS[v], lw=1.5, alpha=0.7)
            ax.plot(x, mean_err, color=COLORS[v], lw=2.5, marker="o", ms=8,
                    markeredgecolor="white", markeredgewidth=1.5, label=LABELS[v])
            ax.annotate(f"{mean_err[-1]:.2f}%", (x[-1], mean_err[-1]), xytext=(10, 0),
                        textcoords="offset points", va="center", color=INK, fontsize=12)
        ax.set_title(name.capitalize(), color=INK, fontsize=16, loc="left")
        ax.set_xticks(x, [f"{int(n)}%" for n in x])
        ax.set_xlim(x[0] - 1.5, x[-1] + 5)
        ax.set_xlabel("missing surface")
        ax.grid(axis="y", color=GRID_COLOR, lw=1)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("mean error in the holes\n(% of the model size)")
    axes[0].set_ylim(bottom=0)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside upper center",
               ncol=3, frameon=False, fontsize=13)
    dest = OUTPUT / "error_vs_damage.png"
    fig.savefig(dest, dpi=300)
    plt.close(fig)
    print(f"figure at {dest.relative_to(ROOT)}")


def grids() -> None:
    """Input, variants and original, one row per level. Each row is viewed from
    the side of the largest hole."""
    import pyvista as pv
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components

    pv.OFF_SCREEN = True

    def poly(V, F):
        return pv.PolyData(V, np.hstack([np.full((len(F), 1), 3), F]).ravel())

    for name in MODELS:
        up = np.array([0.0, 0.0, 1.0]) if MODELS[name][1] else np.array([0.0, 1.0, 0.0])
        v = np.load(DATA / f"{name}_truth.npz")
        truth = Mesh(v["V"], v["F"])
        columns = [("input", "input", "#c97f6f")] + \
                  [(v_, LABELS[v_], COLORS[v_]) for v_ in VARIANTS] + \
                  [("original", "original", "#9ea3a8")]
        height = 1650 if name == "homer" else 1350
        p = pv.Plotter(off_screen=True, shape=(len(LEVELS), len(columns)),
                       window_size=(380 * len(columns), height), border=False)
        for row, fraction in enumerate(LEVELS):
            d = np.load(DATA / f"{name}_{int(fraction * 100)}.npz")
            present = {tuple(sorted(map(tuple, q.tolist())))
                       for q in d["input_V"][d["input_F"]]}
            missing = np.array([tuple(sorted(map(tuple, truth.V[t].tolist()))) not in present
                                for t in truth.F])
            sub = truth.F[missing]
            fe = Mesh(truth.V, sub).face_edges.ravel()
            face_idx = np.repeat(np.arange(len(sub)), 3)
            order = np.argsort(fe, kind="stable")
            fe, face_idx = fe[order], face_idx[order]
            pairs = fe[1:] == fe[:-1]
            graph = sp.coo_matrix((np.ones(pairs.sum()), (face_idx[:-1][pairs], face_idx[1:][pairs])),
                                  shape=(len(sub),) * 2)
            _, comp = connected_components(graph, directed=False)
            largest = np.argmax(np.bincount(comp, weights=Mesh(truth.V, sub).face_areas))
            center = truth.V[sub[comp == largest]].reshape(-1, 3).mean(axis=0)
            horizontal = center - (center @ up) * up
            horizontal /= max(np.linalg.norm(horizontal), 1e-9)
            camera = [tuple(horizontal * 2.9 + up * 0.9), (0, 0, 0), tuple(up)]
            for col, (key, title, color) in enumerate(columns):
                if key == "original":
                    V, F = truth.V, truth.F
                else:
                    V, F = d[f"{key}_V"], d[f"{key}_F"]
                p.subplot(row, col)
                p.add_mesh(poly(V, F), color=color, specular=0.2)
                if row == 0:
                    p.add_text(title, font_size=12, color=INK)
                if col == 0:
                    p.add_text(f"\n\n{int(fraction * 100)}% missing", font_size=11, color=INK)
                p.camera_position = camera
        dest = OUTPUT / f"damage_levels_{name}.png"
        p.screenshot(dest)
        p.close()
        print(f"figure at {dest.relative_to(ROOT)}")


def main() -> None:
    if "--figures" in sys.argv:
        rows = json.loads((OUTPUT / "damage_levels.json").read_text())
    else:
        rows = run()
    plot_errors(rows)
    grids()


if __name__ == "__main__":
    main()
