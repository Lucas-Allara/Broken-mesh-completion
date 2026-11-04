"""Segmentation by graph cut (Section 5 of the paper).

Each tetrahedron of a conforming tetrahedralization is labeled inside or outside
by minimizing the energy of Equation 8: a data term that comes from w and a
smoothness term between neighbors. The output is the set of faces that separate
the two labels.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from .mesh import Mesh
from .tetmesh import TetMesh
from .winding import WindingHierarchy, winding_number

# Symmetric 4-point rule on the tetrahedron (exact up to degree 2), in
# barycentric coordinates. All weigh 1/4.
_A = (5.0 + 3.0 * np.sqrt(5.0)) / 20.0
_B = (5.0 - np.sqrt(5.0)) / 20.0
_BARYCENTRIC_4PT = np.array([
    [_A, _B, _B, _B],
    [_B, _A, _B, _B],
    [_B, _B, _A, _B],
    [_B, _B, _B, _A],
])


def tet_adjacency(mesh: TetMesh) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pairs of tetrahedra that share a face and the area of that face: the
    ``N(i)`` and the ``a_ij`` of Equation 10."""
    tf = mesh.tet_faces.ravel()
    tet_id = np.repeat(np.arange(mesh.n_tets), 4)
    order = np.argsort(tf, kind="stable")
    tet_ord = tet_id[order]

    # Sorted by face, the block of face f starts at the sum of the
    # counts of the previous ones.
    start = np.concatenate([[0], np.cumsum(mesh.face_tet_count)])[:-1]
    interior = np.flatnonzero(mesh.face_tet_count == 2)

    tet_a = tet_ord[start[interior]]
    tet_b = tet_ord[start[interior] + 1]
    return tet_a, tet_b, mesh.face_areas[interior]


def element_winding_number(surf: Mesh, mesh: TetMesh, quadrature: int = 4,
                           hierarchical: bool = True) -> np.ndarray:
    """w averaged over each tetrahedron, at the barycenter (``quadrature=1``)
    or with the 4-point rule."""
    V = mesh.V[mesh.T]
    evaluate = WindingHierarchy(surf) if hierarchical else (lambda P: winding_number(surf, P))

    if quadrature == 1:
        return evaluate(V.mean(axis=1))
    if quadrature != 4:
        raise ValueError("quadrature must be 1 or 4")

    points = np.einsum("qk,mkj->qmj", _BARYCENTRIC_4PT, V)
    return evaluate(points.reshape(-1, 3)).reshape(4, -1).mean(axis=0)


def data_term(w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Equation 9: cost of labeling each element inside and outside."""
    return np.maximum(1.0 - w, 0.0), np.maximum(w, 0.0)


def smoothness_weights(w: np.ndarray, tet_a: np.ndarray, tet_b: np.ndarray,
                       area: np.ndarray, sigma: float) -> np.ndarray:
    """Equation 10: cost of separating two neighbors. It is small where w already
    jumps between them."""
    dif2 = (w[tet_a] - w[tet_b]) ** 2
    return area * np.exp(-dif2 / (2.0 * sigma ** 2))


def estimate_sigma(w: np.ndarray, tet_a: np.ndarray, tet_b: np.ndarray,
                   percentile: float = 85.0) -> float:
    """σ of Equation 10, which the paper does not fix.

    A high percentile of ``|w_i - w_j|`` between neighbors is used. The median does
    not work: almost all pairs are far from the surface and have the same value,
    so it gives a tiny σ and the smoothness term does nothing.
    """
    p = float(np.percentile(np.abs(w[tet_a] - w[tet_b]), percentile))
    return max(p, 1e-6)


def _to_mask(x: np.ndarray, m: int) -> np.ndarray:
    x = np.asarray(x)
    if x.dtype == bool:
        return x
    mask = np.zeros(m, dtype=bool)
    mask[x] = True
    return mask


def apply_hard_constraints(u_inside: np.ndarray, u_outside: np.ndarray,
                           v_weights: np.ndarray,
                           force_inside: np.ndarray | None = None,
                           force_outside: np.ndarray | None = None,
                           ) -> tuple[np.ndarray, np.ndarray]:
    """Fixes the label of some tetrahedra (masks or indices) by setting an
    infinite cost on the opposite side.

    The infinity is larger than the sum of all the costs of the graph, so
    no minimum cut violates a constraint.

    Section 5.2 asks for something weaker (that at least one of the two tetrahedra
    of each input face ends up inside), which is not submodular and the paper
    solves with a heuristic. Fixing only one of the two is solved
    exactly.
    """
    u_inside = np.array(u_inside, dtype=np.float64, copy=True)
    u_outside = np.array(u_outside, dtype=np.float64, copy=True)
    if force_inside is None and force_outside is None:
        return u_inside, u_outside

    m = len(u_inside)
    fi = np.zeros(m, dtype=bool) if force_inside is None else _to_mask(force_inside, m)
    fo = np.zeros(m, dtype=bool) if force_outside is None else _to_mask(force_outside, m)
    if np.any(fi & fo):
        raise ValueError("a tetrahedron cannot be forced both inside and outside")

    infinity = 1.0 + u_inside.sum() + u_outside.sum() + 2.0 * v_weights.sum()
    u_outside[fi] = infinity
    u_inside[fi] = 0.0
    u_inside[fo] = infinity
    u_outside[fo] = 0.0
    return u_inside, u_outside


def hard_constraints_from_surface(surf: Mesh, mesh: TetMesh,
                                  ) -> tuple[np.ndarray, np.ndarray]:
    """For each triangle of ``surf`` that is a face of ``mesh``, marks as inside the
    tetrahedron on the side opposite the normal and as outside the one on the side of the normal.

    The vertices must coincide exactly: on a grid the nearest neighbor
    always exists and would give spurious faces.

    Forcing outside is wrong with interlocking pieces (a face buried inside
    another piece has volume on both sides), which is why ``graphcut_segment`` uses
    only ``force_inside``.
    """
    from scipy.spatial import cKDTree

    d, ids = cKDTree(mesh.V).query(np.asarray(surf.V))
    diagonal = float(np.linalg.norm(mesh.V.max(axis=0) - mesh.V.min(axis=0)))
    matches = d <= 1e-12 * diagonal
    tet_a, tet_b, _ = tet_adjacency(mesh)
    interior = np.flatnonzero(mesh.face_tet_count == 2)
    index = {tuple(face): pos for pos, face in
             enumerate(np.sort(mesh.faces[interior], axis=1))}

    normals = surf.face_normals
    force_inside = np.zeros(mesh.n_tets, dtype=bool)
    force_outside = np.zeros(mesh.n_tets, dtype=bool)

    for k, tri in enumerate(np.asarray(surf.F)):
        if not matches[tri].all():
            continue
        pos = index.get(tuple(sorted(int(ids[v]) for v in tri)))
        if pos is None:
            continue

        a, b = tet_a[pos], tet_b[pos]
        face_center = mesh.V[mesh.faces[interior[pos]]].mean(axis=0)
        center_a = mesh.V[mesh.T[a]].mean(axis=0)
        if np.dot(normals[k], center_a - face_center) > 0:
            outside_t, inside_t = a, b
        else:
            outside_t, inside_t = b, a
        force_inside[inside_t] = True
        force_outside[outside_t] = True

    return force_inside, force_outside


def min_cut_labels(u_inside: np.ndarray, u_outside: np.ndarray,
                   tet_a: np.ndarray, tet_b: np.ndarray,
                   v_weights: np.ndarray, scale: float = 2.0 ** 20) -> np.ndarray:
    """Minimizes Equation 8 with a minimum s-t cut. Returns ``True`` for
    the elements that end up inside.

    Source = inside, sink = outside; ``S→i`` costs ``u_outside(i)``,
    ``i→T`` costs ``u_inside(i)`` and each pair of neighbors carries an edge in
    each direction. ``maximum_flow`` requires integer capacities, so they are scaled.
    """
    from scipy.sparse.csgraph import maximum_flow

    m = len(u_inside)
    S, T = m, m + 1
    n = m + 2

    costs = np.concatenate([u_inside, u_outside, v_weights])
    peak = float(costs.max()) if len(costs) and costs.max() > 0 else 1.0
    k = scale / peak

    rows = np.concatenate([np.full(m, S), np.arange(m), tet_a, tet_b])
    cols = np.concatenate([np.arange(m), np.full(m, T), tet_b, tet_a])
    caps = np.concatenate([u_outside, u_inside, v_weights, v_weights])
    graph = sp.csr_matrix((np.round(caps * k).astype(np.int64), (rows, cols)), shape=(n, n))
    result = maximum_flow(graph, S, T)

    # What is reachable from the source in the residual graph ends up inside.
    residual = (graph - result.flow).tocsr()
    residual.eliminate_zeros()

    reachable = np.zeros(n, dtype=bool)
    reachable[S] = True
    frontier = [S]
    while frontier:
        u = frontier.pop()
        neighbors = residual.indices[residual.indptr[u]:residual.indptr[u + 1]]
        weights = residual.data[residual.indptr[u]:residual.indptr[u + 1]]
        for v, p in zip(neighbors, weights):
            if p > 0 and not reachable[v]:
                reachable[v] = True
                frontier.append(v)

    return reachable[:m]


def graphcut_segment(surf: Mesh, mesh: TetMesh | None = None,
                     gamma: float = 1.0, sigma: float | None = None,
                     quadrature: int = 4,
                     force_inside: np.ndarray | None = None,
                     force_outside: np.ndarray | None = None,
                     hard_constraints: bool = False,
                     heuristics: bool = False,
                     **kwargs) -> dict:
    """The complete method.

    If ``mesh`` is not given, it is built with ``conforming_tet_mesh(surf, **kwargs)``.
    With ``gamma=0`` it gives the same result as the threshold ``w > 0.5``. With
    ``hard_constraints=True`` those of ``hard_constraints_from_surface`` are added
    (inside only). With ``heuristics=True`` the optional constraints of Section 5.2
    (``gsd.constraints``) are applied after the cut.

    Returns a dict with ``w``, ``inside``, ``V``, ``F``, ``mesh`` and ``sigma``.
    """
    if mesh is None:
        from .conforming import conforming_tet_mesh

        mesh, _ = conforming_tet_mesh(surf, **kwargs)

    if hard_constraints:
        fi_auto, _ = hard_constraints_from_surface(surf, mesh)
        force_inside = fi_auto if force_inside is None else (
            _to_mask(force_inside, mesh.n_tets) | fi_auto)

    w = element_winding_number(surf, mesh, quadrature=quadrature)
    tet_a, tet_b, area = tet_adjacency(mesh)

    u_inside, u_outside = data_term(w)
    if sigma is None:
        sigma = estimate_sigma(w, tet_a, tet_b)
    v_weights = gamma * smoothness_weights(w, tet_a, tet_b, area, sigma)

    if force_inside is not None or force_outside is not None:
        u_inside, u_outside = apply_hard_constraints(
            u_inside, u_outside, v_weights, force_inside, force_outside)

    inside = min_cut_labels(u_inside, u_outside, tet_a, tet_b, v_weights)
    if heuristics:
        from .constraints import input_faces, enforce_optional_constraints

        inside = enforce_optional_constraints(mesh, input_faces(surf, mesh), w, inside,
                                              u_inside, u_outside, tet_a, tet_b, v_weights)
    V, F = extract_label_boundary(mesh, inside)
    return {"w": w, "inside": inside, "V": V, "F": F, "mesh": mesh, "sigma": sigma}


def extract_label_boundary(mesh: TetMesh, inside: np.ndarray
                           ) -> tuple[np.ndarray, np.ndarray]:
    """Faces with exactly one inside tetrahedron, oriented outward and
    with the vertices compacted. It includes those on the boundary of the domain, which with the
    CDT of the convex hull are part of the input."""
    tf = mesh.tet_faces.ravel()
    tet = np.repeat(np.arange(mesh.n_tets), 4)
    sel = inside[tet]
    sel &= (np.bincount(tf[sel], minlength=mesh.n_faces) == 1)[tf]
    F = mesh.faces[tf[sel]].copy()

    tet_center = mesh.V[mesh.T[tet[sel]]].mean(axis=1)
    face_center = mesh.V[F].mean(axis=1)
    normal = np.cross(mesh.V[F[:, 1]] - mesh.V[F[:, 0]], mesh.V[F[:, 2]] - mesh.V[F[:, 0]])
    outward = np.einsum("ij,ij->i", normal, face_center - tet_center) > 0
    F[~outward] = F[~outward][:, [0, 2, 1]]

    used, F_compact = np.unique(F, return_inverse=True)
    return mesh.V[used], F_compact.reshape(F.shape).astype(np.int64)
