"""Figures of the poster, without text: the labels are set by poster.tex.

Uses the results saved by demos/demo_damage_levels.py (run it first).

    python poster/poster_figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from gsd.mesh import Mesh  # noqa: E402

DATA = ROOT / "figures" / "level_data"
OUTPUT = Path(__file__).resolve().parent / "figures"
FONTS = (Path.home() / ".local" / "share" / "TinyTeX" / "texmf-dist" / "fonts" / "opentype"
         / "public" / "fira")

COLORS = {"input": "#c97f6f", "method": "#2a78d6", "faired": "#eb6834",
          "symmetry": "#1baf7a", "original": "#9ea3a8"}
COLUMNS = ("input", "method", "faired", "symmetry", "original")
LABELS = {"method": "method of [1]", "faired": "with fairing", "symmetry": "with fairing and symmetry"}
INK = "#1d2433"
INK_SECONDARY = "#4a5160"
GRID_COLOR = "#dcdcd6"
MM = 1 / 25.4


# ------------------------------------------------------------------ renders

def _poly(V, F):
    import pyvista as pv
    return pv.PolyData(np.asarray(V, float), np.hstack([np.full((len(F), 1), 3), F]).ravel())


def _render(layers, cam_pos, win_size, zoom=1.0, edges=False, plane=None) -> Image.Image:
    """``layers``: list of ``(V, F, color)``. Returns RGBA with a transparent background."""
    import pyvista as pv

    pv.OFF_SCREEN = True
    p = pv.Plotter(off_screen=True, window_size=win_size)
    p.set_background("white")
    for V, F, color in layers:
        p.add_mesh(_poly(V, F), color=color, specular=0.25, specular_power=20,
                   show_edges=edges, edge_color="#2b2b2b", line_width=win_size[0] / 700)
    if plane is not None:
        p.add_mesh(plane, color="#3d3d3d", opacity=0.22)
        border = plane.extract_feature_edges(boundary_edges=True, feature_edges=False,
                                             manifold_edges=False)
        p.add_mesh(border, color="#3d3d3d", line_width=win_size[0] / 300)
    p.camera_position = cam_pos
    p.camera.zoom(zoom)
    p.enable_anti_aliasing("ssaa")
    img = Image.fromarray(p.screenshot(return_img=True, transparent_background=True))
    p.close()
    return img


def _crop_together(images, margin=12):
    """Crops all of them with the same box, the union of their contents."""
    boxes = [c for c in (im.getchannel("A").getbbox() for im in images) if c]
    x0 = max(min(c[0] for c in boxes) - margin, 0)
    y0 = max(min(c[1] for c in boxes) - margin, 0)
    x1 = min(max(c[2] for c in boxes) + margin, images[0].width)
    y1 = min(max(c[3] for c in boxes) + margin, images[0].height)
    return [im.crop((x0, y0, x1, y1)) for im in images]


def _row(images, dest: Path) -> None:
    """Pastes the images side by side, all of the same width."""
    images = _crop_together(images)
    row = Image.new("RGBA", (sum(i.width for i in images), images[0].height), (0, 0, 0, 0))
    x = 0
    for im in images:
        row.paste(im, (x, 0))
        x += im.width
    row.save(dest)
    print(f"{dest.relative_to(ROOT)}: {row.size}")


def _load(name: str, level: int) -> dict:
    d = np.load(DATA / f"{name}_{level}.npz")
    v = np.load(DATA / f"{name}_truth.npz")
    meshes = {"input": (d["input_V"], d["input_F"]), "original": (v["V"], v["F"])}
    for k in ("method", "faired", "symmetry"):
        meshes[k] = (d[f"{k}_V"], d[f"{k}_F"])
    return meshes


def _camera_at_hole(meshes, up, dist, cam_height, focus=(0.0, 0.0, 0.0)):
    """Horizontal camera on the side of the largest hole."""
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components

    Ve, Fe = meshes["input"]
    Vo, Fo = meshes["original"]
    present = {tuple(sorted(map(tuple, q.tolist()))) for q in Ve[Fe]}
    missing = np.array([tuple(sorted(map(tuple, Vo[t].tolist()))) not in present for t in Fo])
    sub = Fo[missing]
    hole_mesh = Mesh(Vo, sub)
    fe = hole_mesh.face_edges.ravel()
    face_idx = np.repeat(np.arange(len(sub)), 3)
    order = np.argsort(fe, kind="stable")
    fe, face_idx = fe[order], face_idx[order]
    pairs = fe[1:] == fe[:-1]
    graph = sp.coo_matrix((np.ones(pairs.sum()), (face_idx[:-1][pairs], face_idx[1:][pairs])),
                          shape=(len(sub),) * 2)
    _, comp = connected_components(graph, directed=False)
    largest = np.argmax(np.bincount(comp, weights=hole_mesh.face_areas))
    hole_center = Vo[sub[comp == largest]].reshape(-1, 3).mean(axis=0)
    up = np.asarray(up, float)
    h = hole_center - (hole_center @ up) * up
    h /= max(np.linalg.norm(h), 1e-9)
    focus = np.asarray(focus, float)
    return [tuple(focus + h * dist + up * cam_height), tuple(focus), tuple(up)]


def teaser() -> None:
    """Homer with 10% missing, upper body."""
    m = _load("homer", 10)
    cam = [(0.85, 0.6, 2.4), (0.0, 0.2, 0.0), (0, 1, 0)]
    _row([_render([(*m[k], COLORS[k])], cam, (2000, 2200), zoom=1.25) for k in COLUMNS],
         OUTPUT / "teaser.png")


def results() -> None:
    """Spot and Homer with 22% missing, full body."""
    for name, up, cam_height, win_size in (("spot", (0, 0, 1), 1.0, (1800, 1620)),
                                       ("homer", (0, 1, 0), 0.6, (1440, 2160))):
        m = _load(name, 22)
        cam = _camera_at_hole(m, up, dist=3.2, cam_height=cam_height)
        zoom = 1.05 if name == "spot" else 1.0
        _row([_render([(*m[k], COLORS[k])], cam, win_size, zoom=zoom) for k in COLUMNS],
             OUTPUT / f"{name}_22.png")


def cap() -> None:
    """Close-up of the hole in the neck of Spot: method and fairing."""
    m = _load("spot", 22)
    cam = _camera_at_hole(m, (0, 0, 1), dist=1.45, cam_height=0.4, focus=(-0.1, 0.0, 0.05))
    _row([_render([(*m[k], COLORS[k])], cam, (1500, 1300), edges=True)
          for k in ("method", "faired")], OUTPUT / "cap.png")


def mirror() -> None:
    """Homer with 10% missing: the input, and the input with the plane and the mirrored faces."""
    import pyvista as pv

    from gsd.symmetry import mirror_completion, symmetry_plane

    m = _load("homer", 10)
    input_mesh = Mesh(*m["input"])
    n, d, _ = symmetry_plane(input_mesh)
    mirrored = mirror_completion(input_mesh, Mesh(*m["faired"]), n, d)
    plane = pv.Plane(center=n * d + np.array([0, 0.33, 0.0]), direction=n, i_size=0.75, j_size=0.62)
    cam = [(1.25, 0.55, 2.1), (0.0, 0.3, 0.0), (0, 1, 0)]
    layers = [(*m["input"], COLORS["input"])]
    _row([_render(layers, cam, (1600, 1440), zoom=1.35),
          _render(layers + [(mirrored.V, mirrored.F, COLORS["symmetry"])], cam,
                  (1600, 1440), zoom=1.35, plane=plane)],
         OUTPUT / "mirror.png")


# --------------------------------------------------------------- matplotlib

def _style():
    import matplotlib
    from matplotlib import font_manager

    matplotlib.use("Agg")
    for font_file in ("FiraSans-Regular.otf", "FiraSans-Medium.otf", "FiraSans-SemiBold.otf"):
        font_manager.fontManager.addfont(str(FONTS / font_file))
    matplotlib.rcParams.update({
        "font.family": "Fira Sans", "font.size": 22, "mathtext.fontset": "custom",
        "mathtext.rm": "Fira Sans", "mathtext.it": "Fira Sans:italic",
        "axes.edgecolor": GRID_COLOR, "axes.labelcolor": INK_SECONDARY,
        "xtick.color": INK_SECONDARY, "ytick.color": INK_SECONDARY,
        "mathtext.cal": "Fira Sans", "text.color": INK, "pdf.fonttype": 42,
    })


def w_cut() -> None:
    """w on the plane y = 0 for a closed sphere and for the same sphere without a cap."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    from gsd.shapes import icosphere, punch_hole
    from gsd.winding import sample_grid

    _style()
    w_map = LinearSegmentedColormap.from_list("w", ["#ffffff", "#bcd6f3", "#5d9be3"])
    intact = icosphere(4)
    broken = punch_hole(intact, np.array([0.0, 0.0, 1.0]), 0.6)
    t = np.linspace(0, 2 * np.pi, 2000)
    x, z = np.sin(t), np.cos(t)

    fig, axes = plt.subplots(1, 2, figsize=(205 * MM, 96 * MM), constrained_layout=True)
    for ax, surf, holed in ((axes[0], intact, False), (axes[1], broken, True)):
        w, ext = sample_grid(surf, resolution=320, padding=0.4, axis=1)
        im = ax.imshow(w, extent=ext, origin="lower", cmap=w_map, vmin=0, vmax=1,
                       interpolation="bilinear")
        ax.contour(w, levels=[0.5], extent=ext, colors=INK, linewidths=1.6, linestyles="--")
        border = np.ones_like(t, bool) if not holed else np.hypot(x, z - 1) >= 0.6
        xs, zs = np.where(border, x, np.nan), np.where(border, z, np.nan)
        ax.plot(xs, zs, color=INK, lw=3.2)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    cbar = fig.colorbar(im, ax=axes, shrink=0.85, ticks=[0, 0.5, 1], pad=0.02)
    cbar.ax.tick_params(labelsize=22, length=0)
    cbar.ax.set_yticklabels(["0", "0.5", "1"])
    cbar.outline.set_visible(False)
    cbar.ax.set_title("$w$", fontsize=26, pad=10)
    dest = OUTPUT / "w_cut.pdf"
    fig.savefig(dest, transparent=True)
    plt.close(fig)
    print(dest.relative_to(ROOT))


def error_plot() -> None:
    """Error in the holes against the removed surface, one panel per model."""
    import json

    import matplotlib.pyplot as plt

    _style()
    rows = json.loads((ROOT / "figures" / "damage_levels.json").read_text())
    damage = (0.05, 0.10, 0.22)
    variants = ("method", "faired", "symmetry")
    x = 100 * np.array(damage)

    fig, axes = plt.subplots(1, 2, figsize=(285 * MM, 138 * MM), sharey=True,
                             constrained_layout=True)
    for ax, name in zip(axes, ("spot", "homer")):
        for k, v in enumerate(variants):
            errs = np.array([[f[v] for f in rows if f["model"] == name and f["level"] == n]
                               for n in damage])
            mean_err = errs.mean(axis=1)
            ax.vlines(x + 0.55 * (k - 1), errs.min(axis=1), errs.max(axis=1),
                      color=COLORS[v], lw=3, alpha=0.55)
            ax.plot(x, mean_err, color=COLORS[v], lw=4.5, marker="o", ms=13,
                    markeredgecolor="white", markeredgewidth=2.2, label=LABELS[v])
            ax.annotate(f"{mean_err[-1]:.1f}", (x[-1], mean_err[-1]), xytext=(14, 0),
                        textcoords="offset points", va="center", color=COLORS[v],
                        fontsize=22, fontweight="semibold")
        ax.set_title(name.capitalize(), loc="left", fontsize=28, fontweight="semibold",
                     color=INK)
        ax.set_xticks(x, [f"{int(n)}%" for n in x])
        ax.set_xlim(x[0] - 2, x[-1] + 6)
        ax.set_xlabel("removed surface", fontsize=22)
        ax.grid(axis="y", color=GRID_COLOR, lw=1.2)
        ax.set_axisbelow(True)
        ax.tick_params(length=0, labelsize=21)
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
    axes[0].set_ylabel("mean error in the holes", fontsize=22)
    axes[0].set_ylim(0, 4.8)
    axes[0].yaxis.set_major_formatter(lambda v, _: f"{v:g}%")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside upper center", ncol=3,
               frameon=False, fontsize=22, handlelength=1.6, columnspacing=1.6)
    dest = OUTPUT / "error.pdf"
    fig.savefig(dest, transparent=True)
    plt.close(fig)
    print(dest.relative_to(ROOT))


def main() -> None:
    OUTPUT.mkdir(exist_ok=True)
    requested = sys.argv[1:] or ["teaser", "results", "cap", "mirror", "w_cut", "error_plot"]
    for name in requested:
        globals()[name]()


if __name__ == "__main__":
    main()
