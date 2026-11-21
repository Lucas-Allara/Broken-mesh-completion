"""Figures of the poster, without text: the labels are set by content.tex and layout.tex.

Draws what the C++ programs write, so they have to be built and the experiment run first:

    build/gsd_experiment . results/damage_levels.json --meshes results/meshes
    python poster/poster_figures.py [teaser] [results] [cap] [mirror] [w_cut] [error_plot]

Without arguments it writes every figure into poster/figures:
  teaser      Homer with the holes of the 10% level, as the input, completed by each stage and
              as the original (teaser.png);
  results     Spot and Homer with the holes of the 22% level, likewise (spot_22.png, homer_22.png);
  cap         the hole in the neck of Spot closed by the method and by fairing, edges drawn
              (cap.png);
  mirror      Homer with the symmetry plane and the mirrored faces that fall in its holes
              (mirror.png);
  w_cut       the winding number on a slice of a closed sphere and of the same sphere with a
              hole, which build/gsd_slice computes (w_cut.pdf);
  error_plot  the error in the holes against the removed surface (error.pdf).
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
MESHES = ROOT / "results" / "meshes"
NUMBERS = ROOT / "results" / "damage_levels.json"
SLICE = ROOT / "build" / "gsd_slice"
OUTPUT = Path(__file__).resolve().parent / "figures"
SEED = 3  # the set of holes that the renders show

# The palette of the poster, around the burgundy of UTDT. Teal marks the method of [1], amber
# fairing, burgundy fairing with symmetry, light pink the input and gray the original.
COLORS = {"input": "#e8bfcc", "method": "#008c9a", "faired": "#d9922e",
          "symmetry": "#820a32", "original": "#b4b4b8"}
# The renders take a lighter burgundy, since #820a32 is too dark for the shading to show the shape.
MESH_COLORS = dict(COLORS, symmetry="#a8284f")
COLUMNS = ("input", "method", "faired", "symmetry", "original")
LABELS = {"method": "method of [1]", "faired": "with fairing", "symmetry": "with fairing and symmetry"}
INK = "#2b2b2d"
INK_SECONDARY = "#5e5e63"
GRID_COLOR = "#e4dde0"
MM = 1 / 25.4


# ------------------------------------------------------------------ data

def _load_obj(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Vertices and triangles of an OBJ written by the C++ programs."""
    if not path.exists():
        raise SystemExit(f"{path.relative_to(ROOT)} is missing: run "
                         "build/gsd_experiment . results/damage_levels.json --meshes results/meshes first")
    V, F = [], []
    with open(path) as lines:
        for line in lines:
            if line.startswith("v "):
                V.append([float(x) for x in line.split()[1:4]])
            elif line.startswith("f "):
                F.append([int(x) - 1 for x in line.split()[1:4]])
    return np.array(V, dtype=float), np.array(F, dtype=np.int64)


def _load(name: str, level: int) -> dict:
    """The five meshes of one case of the experiment, and its mirrored faces."""
    meshes = {k: _load_obj(MESHES / f"{name}_{level}_{SEED}_{k}.obj")
              for k in ("input", "method", "faired", "symmetry", "mirrored")}
    meshes["original"] = _load_obj(MESHES / f"{name}_original.obj")
    return meshes


def _rows() -> list[dict]:
    if not NUMBERS.exists():
        raise SystemExit(f"{NUMBERS.relative_to(ROOT)} is missing: run build/gsd_experiment first")
    return json.loads(NUMBERS.read_text())


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


def _camera_at_hole(meshes, up, dist, cam_height, focus=(0.0, 0.0, 0.0)):
    """Horizontal camera on the side of the largest hole."""
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components

    Ve, Fe = meshes["input"]
    Vo, Fo = meshes["original"]
    present = {tuple(sorted(map(tuple, q.tolist()))) for q in Ve[Fe]}
    missing = np.array([tuple(sorted(map(tuple, Vo[t].tolist()))) not in present for t in Fo])
    sub = Fo[missing]
    # The removed faces that share an edge belong to the same hole.
    edges = np.sort(np.stack([sub[:, [1, 2]], sub[:, [2, 0]], sub[:, [0, 1]]], axis=1).reshape(-1, 2), axis=1)
    edge_id = np.unique(edges, axis=0, return_inverse=True)[1].ravel()
    face = np.repeat(np.arange(len(sub)), 3)
    order = np.argsort(edge_id, kind="stable")
    edge_id, face = edge_id[order], face[order]
    pairs = edge_id[1:] == edge_id[:-1]
    graph = sp.coo_matrix((np.ones(pairs.sum()), (face[:-1][pairs], face[1:][pairs])),
                          shape=(len(sub),) * 2)
    _, comp = connected_components(graph, directed=False)
    T = Vo[sub]
    areas = 0.5 * np.linalg.norm(np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]), axis=1)
    largest = np.argmax(np.bincount(comp, weights=areas))
    hole_center = T[comp == largest].reshape(-1, 3).mean(axis=0)
    up = np.asarray(up, float)
    h = hole_center - (hole_center @ up) * up
    h /= max(np.linalg.norm(h), 1e-9)
    focus = np.asarray(focus, float)
    return [tuple(focus + h * dist + up * cam_height), tuple(focus), tuple(up)]


def teaser() -> None:
    """Homer with 10% missing, upper body."""
    m = _load("homer", 10)
    cam = [(0.85, 0.6, 2.4), (0.0, 0.2, 0.0), (0, 1, 0)]
    _row([_render([(*m[k], MESH_COLORS[k])], cam, (2000, 2200), zoom=1.25) for k in COLUMNS],
         OUTPUT / "teaser.png")


def results() -> None:
    """Spot and Homer with 22% missing, full body."""
    for name, up, cam_height, win_size in (("spot", (0, 0, 1), 1.0, (1800, 1620)),
                                           ("homer", (0, 1, 0), 0.6, (1440, 2160))):
        m = _load(name, 22)
        cam = _camera_at_hole(m, up, dist=3.2, cam_height=cam_height)
        zoom = 1.05 if name == "spot" else 1.0
        _row([_render([(*m[k], MESH_COLORS[k])], cam, win_size, zoom=zoom) for k in COLUMNS],
             OUTPUT / f"{name}_22.png")


def cap() -> None:
    """Close-up of the hole in the neck of Spot: method and fairing."""
    m = _load("spot", 22)
    cam = _camera_at_hole(m, (0, 0, 1), dist=1.45, cam_height=0.4, focus=(-0.1, 0.0, 0.05))
    _row([_render([(*m[k], MESH_COLORS[k])], cam, (1500, 1300), edges=True)
          for k in ("method", "faired")], OUTPUT / "cap.png")


def mirror() -> None:
    """Homer with 10% missing: the input, and the input with the plane and the mirrored faces."""
    import pyvista as pv

    m = _load("homer", 10)
    case = next(r for r in _rows() if (r["model"], round(100 * r["level"]), r["seed"]) == ("homer", 10, SEED))
    n, d = np.array(case["plane"][:3]), case["plane"][3]
    if n[0] < 0:  # the same plane, with its normal towards +x
        n, d = -n, -d
    plane = pv.Plane(center=n * d + np.array([0, 0.33, 0.0]), direction=n, i_size=0.75, j_size=0.62)
    cam = [(1.25, 0.55, 2.1), (0.0, 0.3, 0.0), (0, 1, 0)]
    layers = [(*m["input"], MESH_COLORS["input"])]
    _row([_render(layers, cam, (1600, 1440), zoom=1.35),
          _render(layers + [(*m["mirrored"], MESH_COLORS["symmetry"])], cam,
                  (1600, 1440), zoom=1.35, plane=plane)],
         OUTPUT / "mirror.png")


# --------------------------------------------------------------- matplotlib

def _font_dir() -> Path:
    """Folder of the Montserrat fonts of the TeX installation: where kpsewhich finds them, or the
    TinyTeX tree in the home directory. Without them matplotlib would fall back to its default
    font without saying so."""
    import shutil

    tool = shutil.which("kpsewhich")
    if tool:
        found = subprocess.run([tool, "Montserrat-Regular.otf"], capture_output=True, text=True).stdout.strip()
        if found:
            return Path(found).parent
    for tree in (Path.home() / ".local" / "share" / "TinyTeX", Path.home() / ".TinyTeX"):
        folder = tree / "texmf-dist" / "fonts" / "opentype" / "public" / "montserrat"
        if any(folder.glob("Montserrat-*.otf")):
            return folder
    raise SystemExit("Montserrat was not found (tlmgr install montserrat)")


def _style():
    import matplotlib
    from matplotlib import font_manager

    matplotlib.use("Agg")
    for font_file in sorted(_font_dir().glob("Montserrat-*.otf")):
        font_manager.fontManager.addfont(str(font_file))
    matplotlib.rcParams.update({
        "font.family": "Montserrat", "font.size": 22, "mathtext.fontset": "custom",
        "mathtext.rm": "Montserrat", "mathtext.it": "Montserrat:italic",
        "axes.edgecolor": GRID_COLOR, "axes.labelcolor": INK_SECONDARY,
        "xtick.color": INK_SECONDARY, "ytick.color": INK_SECONDARY,
        "mathtext.cal": "Montserrat", "text.color": INK, "pdf.fonttype": 42,
    })


def _slice(hole: float) -> tuple[np.ndarray, tuple[float, ...]]:
    """w on the plane y = 0 of the unit sphere without a cap of that radius, from gsd_slice."""
    if not SLICE.exists():
        raise SystemExit(f"{SLICE.relative_to(ROOT)} is missing: build the C++ programs first")
    with tempfile.TemporaryDirectory() as folder:
        out = Path(folder) / "slice.txt"
        subprocess.run([str(SLICE), "--sphere", str(hole), str(out)], check=True, stdout=subprocess.DEVNULL)
        with open(out) as lines:
            extent = tuple(float(x) for x in lines.readline().split()[1:])
            return np.loadtxt(lines), extent


def w_cut() -> None:
    """w on the plane y = 0 for a closed sphere and for the same sphere without a cap."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    _style()
    # White up to 0.12, so that the edge of the image, where w is about 0.08, does not show on the page.
    w_map = LinearSegmentedColormap.from_list("w", [(0, "#ffffff"), (0.12, "#ffffff"), (0.5, "#e8bfcc"), (1, "#c9748f")])
    hole_radius = 0.6
    t = np.linspace(0, 2 * np.pi, 2000)
    x, z = np.sin(t), np.cos(t)

    fig, axes = plt.subplots(1, 2, figsize=(205 * MM, 84 * MM), constrained_layout=True)
    for ax, hole in ((axes[0], 0.0), (axes[1], hole_radius)):
        w, ext = _slice(hole)
        im = ax.imshow(w, extent=ext, origin="lower", cmap=w_map, vmin=0, vmax=1,
                       interpolation="bilinear")
        ax.contour(w, levels=[0.5], extent=ext, colors=INK, linewidths=1.6, linestyles="--")
        border = np.hypot(x, z - 1) >= hole
        xs, zs = np.where(border, x, np.nan), np.where(border, z, np.nan)
        ax.plot(xs, zs, color=INK, lw=3.2)
        ax.set_aspect("equal")
        ax.set_ylim(-1.25, ext[3])  # the band below the sphere is empty
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    cbar = fig.colorbar(im, ax=axes, shrink=0.85, ticks=[0, 0.5, 1], pad=0.02)
    cbar.ax.tick_params(labelsize=22, length=0)
    cbar.ax.set_yticklabels(["0", "0.5", "1"])
    cbar.outline.set_edgecolor(INK_SECONDARY)
    cbar.outline.set_linewidth(1.0)
    cbar.ax.set_title("$w$", fontsize=26, pad=10)
    dest = OUTPUT / "w_cut.pdf"
    fig.savefig(dest, transparent=True, dpi=300)  # the dpi of the two images of w inside the PDF
    plt.close(fig)
    print(dest.relative_to(ROOT))


def error_plot() -> None:
    """Error in the holes against the removed surface, one panel per model. The dotted line is
    the floor of the measure, the original measured against itself."""
    import matplotlib.pyplot as plt

    _style()
    rows = _rows()
    damage = (0.05, 0.10, 0.22)
    variants = ("method", "faired", "symmetry")
    x = 100 * np.array(damage)

    fig, axes = plt.subplots(1, 2, figsize=(315 * MM, 106 * MM), sharey=True,
                             constrained_layout=True)
    for ax, name in zip(axes, ("spot", "homer")):
        floor = np.mean([f["floor"] for f in rows if f["model"] == name])
        ax.axhline(floor, color=INK_SECONDARY, lw=2.4, ls=(0, (1, 2.4)), zorder=1)
        for k, v in enumerate(variants):
            errs = np.array([[f[v] for f in rows if f["model"] == name and f["level"] == n]
                             for n in damage])
            mean_err = errs.mean(axis=1)
            ax.vlines(x + 0.55 * (k - 1), errs.min(axis=1), errs.max(axis=1),
                      color=COLORS[v], lw=3, alpha=0.55)
            ax.plot(x, mean_err, color=COLORS[v], lw=4.5, marker="o", ms=13,
                    markeredgecolor="white", markeredgewidth=2.2, label=LABELS[v])
            ax.annotate(f"{mean_err[-1]:.1f}", (x[-1], mean_err[-1]), xytext=(14, 0),
                        textcoords="offset points", va="center", color=INK,
                        fontsize=17, fontweight="semibold")
        ax.set_title(name.capitalize(), loc="left", fontsize=20, fontweight="semibold",
                     color=INK)
        ax.set_xticks(x, [f"{int(n)}%" for n in x])
        ax.set_xlim(x[0] - 2, x[-1] + 6)
        ax.set_xlabel("removed surface (target)", fontsize=17)
        ax.grid(axis="y", color=GRID_COLOR, lw=1.2)
        ax.set_axisbelow(True)
        ax.tick_params(length=0, labelsize=17)
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
    axes[0].set_ylabel("mean error in the holes", fontsize=17)
    axes[0].set_ylim(0, 4.8)
    axes[0].yaxis.set_major_formatter(lambda v, _: f"{v:g}%")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside upper center", ncol=3,
               frameon=False, fontsize=17, handlelength=1.6, columnspacing=1.6)
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
