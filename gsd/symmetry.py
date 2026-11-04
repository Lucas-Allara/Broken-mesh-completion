"""Filling the holes by mirroring the input across its symmetry plane.

Fairing cannot invent detail: if a hole takes away the nose, the
cap stays smooth. If the model is symmetric, what is missing on one side is usually
on the other.

1. The plane that minimizes the distance from the mirrored surface to the
   original is sought, ignoring the 20% of points that match worst (the holes).
2. The mirrored faces that fall where the input has no surface and that are
   connected with some face close to the fill of a first pass are taken.
   The second condition avoids adding pieces that are not symmetric in the pose
   (an arm raised higher than the other).
3. The reconstruction is run again with those faces added to the input, without
   stitching them.

If the residual of the plane exceeds half an edge, the model is taken as not
symmetric and nothing is done.
"""

from __future__ import annotations

import numpy as np

from .mesh import Mesh


def _samples(V: np.ndarray, F: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """``k`` points uniformly distributed on the surface, by area."""
    T = V[F]
    area = 0.5 * np.linalg.norm(np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]), axis=1)
    f = rng.choice(len(F), size=k, p=area / area.sum())
    u, v = rng.random(k), rng.random(k)
    outside = u + v > 1
    u[outside], v[outside] = 1 - u[outside], 1 - v[outside]
    return T[f, 0] + u[:, None] * (T[f, 1] - T[f, 0]) + v[:, None] * (T[f, 2] - T[f, 0])


def _reflect(P: np.ndarray, n: np.ndarray, d: float) -> np.ndarray:
    """Reflection of ``P`` across the plane ``n·x = d``."""
    return P - 2.0 * (P @ n - d)[:, None] * n


def _normal(theta: float, phi: float) -> np.ndarray:
    return np.array([np.cos(theta) * np.cos(phi), np.sin(theta) * np.cos(phi), np.sin(phi)])


def symmetry_plane(surf: Mesh, n_samples: int = 4000, trim: float = 0.8,
                   seed: int = 0) -> tuple[np.ndarray, float, float]:
    """Symmetry plane ``n·x = d`` of ``surf``. Returns ``(n, d, residual)``,
    with the residual the mean distance from the mirrored surface to the original
    over the fraction ``trim`` of points that match best."""
    from scipy.optimize import minimize
    from scipy.spatial import cKDTree

    rng = np.random.default_rng(seed)
    V, F = np.asarray(surf.V), np.asarray(surf.F)
    P = _samples(V, F, n_samples, rng)
    tree = cKDTree(_samples(V, F, min(max(50_000, 10 * len(F)), 300_000), rng))
    cut = int(trim * len(P))

    def cost(x: np.ndarray) -> float:
        d = tree.query(_reflect(P, _normal(x[0], x[1]), x[2]))[0]
        return float(np.mean(np.partition(d, cut)[:cut]))

    center = P.mean(axis=0)
    _, _, axes = np.linalg.svd(P - center)
    best = None
    for n0 in axes:
        x0 = np.array([np.arctan2(n0[1], n0[0]), np.arcsin(np.clip(n0[2], -1, 1)), center @ n0])
        r = minimize(cost, x0, method="Nelder-Mead",
                     options=dict(xatol=1e-5, fatol=1e-7, maxiter=400))
        if best is None or r.fun < best.fun:
            best = r
    return _normal(best.x[0], best.x[1]), float(best.x[2]), float(best.fun)


def mirror_completion(surf: Mesh, filled: Mesh, n: np.ndarray, d: float,
                      far: float = 2.0, near: float = 0.06, seed: int = 0) -> Mesh:
    """Faces of ``surf`` mirrored that fall into its holes.

    ``filled`` is a first reconstruction. The mirrored faces farther than ``far``
    edges from the input and connected (through faces of that kind) with
    some face closer than ``near`` times the side of the box of the fill are taken.
    Mirroring reverses the orientation, so they are flipped.
    """
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree

    rng = np.random.default_rng(seed)
    h = float(np.median(surf.edge_lengths))
    side = float((filled.V.max(axis=0) - filled.V.min(axis=0)).max())

    VR = _reflect(np.asarray(surf.V), n, d)
    FR = np.asarray(surf.F)[:, ::-1]
    centers = VR[FR].mean(axis=1)
    input_tree = cKDTree(_samples(surf.V, surf.F, max(200_000, 20 * surf.n_faces), rng))
    in_void = input_tree.query(centers)[0] > far * h
    fill_tree = cKDTree(_samples(filled.V, filled.F, max(200_000, 20 * filled.n_faces), rng))
    seeds = in_void & (fill_tree.query(centers)[0] < near * side)

    # Components of the mirrored faces in the void, joined by edges.
    mirrored_mesh = Mesh(VR, FR)
    fe = mirrored_mesh.face_edges[in_void].ravel()
    face = np.repeat(np.flatnonzero(in_void), 3)
    order = np.argsort(fe, kind="stable")
    fe, face = fe[order], face[order]
    pairs = fe[1:] == fe[:-1]
    m = len(FR)
    graph = sp.coo_matrix((np.ones(pairs.sum()), (face[:-1][pairs], face[1:][pairs])), shape=(m, m))
    _, comp = connected_components(graph, directed=False)
    keep = in_void & np.isin(comp, np.unique(comp[seeds]))

    used, F = np.unique(FR[keep], return_inverse=True)
    return Mesh(VR[used], F.reshape(-1, 3))


def reconstruct_with_symmetry(surf: Mesh, symmetry: bool = True, max_residual: float = 0.5,
                              gamma: float = 40.0, quadrature: int = 1,
                              **kwargs) -> dict:
    """Segmentation, fairing and, if the model is symmetric, a second
    pass with the input completed by reflection.

    ``**kwargs`` goes to ``conforming_tet_mesh``. Returns a dict with ``method`` and
    ``faired`` (``(V, F)`` pairs of the first pass), ``plane``
    (``(n, d, residual)``), ``mirrored`` and ``symmetry``; the last ones are
    ``None`` if symmetry was not used.
    """
    from .conforming import conforming_tet_mesh
    from .fairing import fair_filled_patches
    from .segmentation import graphcut_segment

    def run_pass(target: Mesh) -> tuple[tuple, tuple]:
        mesh, _ = conforming_tet_mesh(target, **kwargs)
        r = graphcut_segment(target, mesh=mesh, gamma=gamma, quadrature=quadrature,
                             heuristics=True)
        return (r["V"], r["F"]), fair_filled_patches(r["V"], r["F"], target)

    segmented, faired = run_pass(surf)
    result = {"method": segmented, "faired": faired, "plane": None, "symmetry": None,
              "mirrored": None}
    if not symmetry:
        return result

    n, d, residual = symmetry_plane(surf)
    result["plane"] = (n, d, residual)
    if residual > max_residual * float(np.median(surf.edge_lengths)):
        return result

    mirrored = mirror_completion(surf, Mesh(*faired), n, d)
    if mirrored.n_faces == 0:
        return result
    augmented = Mesh(np.vstack([surf.V, mirrored.V]),
                     np.vstack([surf.F, mirrored.F + surf.n_vertices]))
    _, result["symmetry"] = run_pass(augmented)
    result["mirrored"] = mirrored
    return result
