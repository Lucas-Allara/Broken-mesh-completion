"""Tetrahedralization with every input triangle as a face (Section 5 of the
paper). It uses TetGen.

TetGen aborts the process if two triangles of the constraint intersect. That is
why those triangles are detected beforehand and subdivided; the pieces that still
intersect are left out of the constraint (but they still count for w).

The two unconstrained alternatives are also here: Delaunay on the same points and
a regular grid.
"""

from __future__ import annotations

import numpy as np

from .mesh import Mesh
from .tetmesh import TetMesh, drop_degenerate_tets, tet_grid

# Faces of the bounding box, oriented outward.
_BOX_F = np.array([
    [0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5],
    [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6],
    [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3],
])


def _orient3d(a, b, c, d) -> np.ndarray:
    """Sign of the volume of ``abcd`` row by row, with 0 if it is of the order of
    rounding. Without the tolerance, neighboring faces (which share vertices)
    would appear to intersect."""
    u, v, w = b - a, c - a, d - a
    vol = np.einsum("ij,ij->i", u, np.cross(v, w))
    scale = np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1) * np.linalg.norm(w, axis=1)
    return np.where(np.abs(vol) <= 1e-10 * scale, 0.0, np.sign(vol))


def _edge_crosses(P, Q, A, B, C) -> np.ndarray:
    """Whether the segment ``PQ`` crosses the triangle ``ABC`` (its boundary included)."""
    sides = _orient3d(A, B, C, P) * _orient3d(A, B, C, Q)
    t1, t2, t3 = _orient3d(P, Q, A, B), _orient3d(P, Q, B, C), _orient3d(P, Q, C, A)
    inside = ((t1 >= 0) & (t2 >= 0) & (t3 >= 0)) | ((t1 <= 0) & (t2 <= 0) & (t3 <= 0))
    return (sides < 0) & inside


def self_intersecting_faces(surf: Mesh) -> np.ndarray:
    """Mask of the triangles that intersect some other one.

    Two non-coplanar triangles intersect if an edge of one crosses the
    other. Overlapping coplanar ones are not detected.
    """
    from scipy.spatial import cKDTree

    T = np.asarray(surf.V)[np.asarray(surf.F)]
    mask = np.zeros(len(T), dtype=bool)
    if len(T) < 2:
        return mask

    # Candidate pairs: centers closer than the sum of the radii. They are searched
    # by radius class so that a large triangle does not make the search
    # quadratic.
    centers = T.mean(axis=1)
    radii = np.linalg.norm(T - centers[:, None], axis=2).max(axis=1)
    radii = np.maximum(radii, 1e-12 * radii.max())
    size_class = np.floor(np.log2(radii / radii.min())).astype(np.int64)
    groups = [np.flatnonzero(size_class == k) for k in np.unique(size_class)]
    trees = [cKDTree(centers[g]) for g in groups]
    r_max = [float(radii[g].max()) for g in groups]

    pairs = []
    for a in range(len(groups)):
        for b in range(a, len(groups)):
            d = trees[a].sparse_distance_matrix(trees[b], r_max[a] + r_max[b],
                                                output_type="ndarray")
            i, j = groups[a][d["i"]], groups[b][d["j"]]
            pair = np.column_stack([i, j])
            pairs.append(pair if a != b else pair[i < j])
    pairs = np.concatenate(pairs)
    i, j = pairs[:, 0], pairs[:, 1]

    lo, hi = T.min(axis=1), T.max(axis=1)
    overlap = np.all((lo[i] <= hi[j]) & (lo[j] <= hi[i]), axis=1)
    i, j = i[overlap], j[overlap]

    crossing = np.zeros(len(i), dtype=bool)
    for k in range(3):
        s = (k + 1) % 3
        crossing |= _edge_crosses(T[i, k], T[i, s], T[j, 0], T[j, 1], T[j, 2])
        crossing |= _edge_crosses(T[j, k], T[j, s], T[i, 0], T[i, 1], T[i, 2])

    mask[i[crossing]] = True
    mask[j[crossing]] = True
    return mask


def _red_green_refine(V: np.ndarray, F: np.ndarray, marked: np.ndarray
                      ) -> tuple[np.ndarray, np.ndarray]:
    """Splits the marked faces into four without leaving hanging vertices: the
    neighbors with one split edge are bisected, and the ones left with two or more
    are split into four as well."""
    tri_mesh = Mesh(V, F)
    marked = marked.copy()
    while True:
        split = np.zeros(tri_mesh.n_edges, dtype=bool)
        split[tri_mesh.face_edges[marked].ravel()] = True
        new = ~marked & (split[tri_mesh.face_edges].sum(axis=1) >= 2)
        if not new.any():
            break
        marked |= new
    fine = tri_mesh.split_edges(split)
    return fine.V, fine.F.astype(np.int64)


def _box(surf: Mesh, margin: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Corners of the bounding box enlarged by ``margin`` times its longest side,
    and its extremes."""
    lo, hi = surf.bounding_box()
    pad = margin * float((hi - lo).max())
    lo, hi = lo - pad, hi + pad
    corners = np.array([[x, y, z] for x in (lo[0], hi[0])
                        for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    return corners, lo, hi


def conforming_tet_mesh(surf: Mesh, max_far_edge: float | None = None, margin: float = 0.22,
                        quality: float = 1.6, refine_intersections: int = 2
                        ) -> tuple[TetMesh, np.ndarray]:
    """Tetrahedralization with the triangles of ``surf`` as faces.

    Returns ``(mesh, ids)``, with ``ids[k]`` the vertex of the mesh that
    corresponds to vertex ``k`` of the input.

    By default it is the one in the paper: the CDT of the convex hull of the input,
    without new points. If ``max_far_edge`` is given, the box enlarged by ``margin``
    is tetrahedralized and TetGen adds points to bound the quality and the
    length of the edges far from the surface. ``refine_intersections`` is how many
    times the intersecting faces are subdivided; if TetGen rejects the
    result, one fewer time is tried.
    """
    import tetgen
    from scipy.spatial import cKDTree

    if max_far_edge is None:
        # p: PLC, YY: no new points on the input faces, c: keeps the
        # convex hull.
        corners, box_faces, switches = np.empty((0, 3)), np.empty((0, 3), dtype=np.int64), "pYYcQ"
    else:
        corners, _, _ = _box(surf, margin)
        box_faces = _BOX_F
        # q: quality, a: maximum volume.
        switches = f"pq{quality}YYa{max_far_edge ** 3 / 6.0:.12f}Q"

    # The faces that do not intersect go to the constraint. Subdividing does not
    # change the indices of the input vertices (the new ones go at the end).
    V, F = np.asarray(surf.V), np.asarray(surf.F)
    crossing = self_intersecting_faces(surf)
    attempts = [(V, F[~crossing])]
    for _ in range(refine_intersections):
        if not crossing.any():
            break
        V, F = _red_green_refine(V, F, crossing)
        crossing = self_intersecting_faces(Mesh(V, F))
        attempts.append((V, F[~crossing]))

    for k, (V, constrained) in enumerate(reversed(attempts)):
        try:
            result = tetgen.TetGen(np.vstack([corners, V]),
                                   np.vstack([box_faces, constrained + len(corners)])
                                   ).tetrahedralize(switches=switches)
            break
        except RuntimeError:
            if k == len(attempts) - 1:
                raise
    tet_mesh = TetMesh(*drop_degenerate_tets(np.asarray(result[0]), np.asarray(result[1])))
    _, ids = cKDTree(tet_mesh.V).query(np.asarray(surf.V))
    return tet_mesh, ids.astype(np.int64)


def delaunay_tet_mesh(surf: Mesh, margin: float = 0.22) -> TetMesh:
    """Delaunay of the input vertices and the corners of the box, without requiring
    the input triangles to be faces."""
    from scipy.spatial import Delaunay

    corners, _, _ = _box(surf, margin)
    points = np.vstack([corners, surf.V])
    return TetMesh(*drop_degenerate_tets(points, Delaunay(points).simplices))


def grid_tet_mesh(surf: Mesh, n: int = 32, margin: float = 0.22) -> TetMesh:
    """Regular grid of ``n`` cubes per side that covers the box of ``surf``."""
    _, lo, hi = _box(surf, margin)
    return tet_grid(n, size=float((hi - lo).max()), center=0.5 * (lo + hi))
