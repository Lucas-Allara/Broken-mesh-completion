"""Fairing of the caps added by the segmentation.

Where there was a hole, the output surface goes through faces of the
tetrahedralization and ends up wrinkled. Here the points that lie on the
input are kept fixed and those of the caps are relocated so that they cancel the
tri-Laplacian (``Δ³x = 0``), which blends with what surrounds the hole in position,
tangent and curvature. Before that, the connectivity of the caps is fixed with
Delaunay flips and a subdivision.

The input faces are not touched, nor are the tetrahedral mesh and the labels:
it is a post-process of the surface.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from .mesh import Mesh


def fair_filled_patches(V: np.ndarray, F: np.ndarray, surf: Mesh,
                        refine: int = 2, rounds: int = 3, order: int = 3
                        ) -> tuple[np.ndarray, np.ndarray]:
    """Fairs the caps of ``(V, F)`` keeping fixed what lies on ``surf``.

    ``refine`` is how many times the caps are subdivided (with three it already
    gets worse), ``rounds`` how many fairing and flip passes are done per level
    and ``order`` the power of the Laplacian (2 is the thin plate).
    """
    V = np.asarray(V, dtype=np.float64)
    F = np.asarray(F, dtype=np.int64)
    tol = 1e-9 * float(np.linalg.norm(surf.V.max(axis=0) - surf.V.min(axis=0)))

    fixed = _on_input(V, surf, tol)
    is_input = fixed[F].all(axis=1) & _on_input(V[F].mean(axis=1), surf, tol)
    if is_input.all():
        return V.copy(), F.copy()

    for level in range(refine + 1):
        if level > 0:
            V, F, is_input, fixed = _split(Mesh(V, F), is_input, fixed)
        for _ in range(rounds):
            V = _smooth(V, F, fixed, order)
            F = _flips(V, F, is_input)
            F, is_input, fixed = _remove_valence_3(F, is_input, fixed)
            F, is_input, fixed = _collapse_needles(V, F, is_input, fixed)
    V = _smooth(V, F, fixed, order)

    used, F = np.unique(F, return_inverse=True)
    return V[used], F.reshape(-1, 3)


def _split(tri_mesh: Mesh, is_input: np.ndarray, fixed: np.ndarray):
    """Splits at their midpoint all the edges of the caps."""
    to_split = np.ones(tri_mesh.n_edges, dtype=bool)
    to_split[tri_mesh.face_edges[is_input].ravel()] = False
    fine = tri_mesh.split_edges(to_split)
    fixed = np.concatenate([fixed, np.zeros(fine.n_vertices - tri_mesh.n_vertices, dtype=bool)])
    # split_edges puts first, in order, the faces that were not split, and
    # all the input faces are among them.
    whole = ~to_split[tri_mesh.face_edges].any(axis=1)
    is_input = np.concatenate([is_input[whole],
                               np.zeros(fine.n_faces - whole.sum(), dtype=bool)])
    return fine.V, fine.F.astype(np.int64), is_input, fixed


def _on_input(P: np.ndarray, surf: Mesh, tol: float) -> np.ndarray:
    """Which points of ``P`` lie on some triangle of ``surf``.

    It includes the midpoints that ``conforming_tet_mesh`` adds when subdividing
    intersecting faces.
    """
    from scipy.spatial import cKDTree

    T = surf.V[surf.F]
    centers = T.mean(axis=1)
    radius = float(np.linalg.norm(T - centers[:, None], axis=2).max())
    near = cKDTree(centers).query_ball_point(P, radius + tol)
    i = np.repeat(np.arange(len(P)), [len(c) for c in near])
    j = np.concatenate([np.asarray(c, dtype=np.int64) for c in near]) if len(i) else i

    a, b, c = T[j, 0], T[j, 1], T[j, 2]
    n = np.cross(b - a, c - a)
    area2 = np.linalg.norm(n, axis=1)
    ok = area2 > 0
    n[ok] /= area2[ok, None]
    q = P[i]
    on_plane = np.abs(np.einsum("ij,ij->i", q - a, n)) <= tol
    tol_b = tol * (np.linalg.norm(b - a, axis=1) + np.linalg.norm(c - a, axis=1))
    within = np.ones(len(i), dtype=bool)
    for u, v in ((a, b), (b, c), (c, a)):
        within &= np.einsum("ij,ij->i", np.cross(v - u, q - u), n) >= -tol_b
    on = np.zeros(len(P), dtype=bool)
    on[i[ok & on_plane & within]] = True
    return on


# How far a cap may stick out of the box of its anchor vertices, as a
# fraction of the diagonal of that box, before its order is lowered.
_MARGIN = 0.15


def _smooth(V: np.ndarray, F: np.ndarray, fixed: np.ndarray, degree: int) -> np.ndarray:
    """Solves ``L^degree x = 0`` at the free vertices, with ``L`` the
    uniform Laplacian and the fixed ones as boundary.

    With degree 2 or 3 a cap can shoot far away and leave a spike. Each cap
    that leaves the box of its anchors (plus ``_MARGIN``) is solved again
    with one degree less; with degree 1 that does not happen (maximum principle).
    The free pieces that do not touch any fixed vertex stay where they are.
    """
    from scipy.sparse.csgraph import connected_components
    from scipy.sparse.linalg import spsolve

    e = Mesh(V, F).edges
    n = len(V)
    A = sp.coo_matrix((np.ones(2 * len(e)), (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])),
                      shape=(n, n)).tocsr()
    L = sp.diags(np.asarray(A.sum(axis=1)).ravel()) - A

    fixed = fixed.copy()
    free = ~fixed
    _, comp = connected_components(A[free][:, free], directed=False)
    touches_fixed = np.zeros(comp.max() + 1 if len(comp) else 0, dtype=bool)
    fixed_neighbors = np.asarray(A[free][:, fixed].sum(axis=1)).ravel() > 0
    touches_fixed[comp[fixed_neighbors]] = True
    fixed[np.flatnonzero(free)[~touches_fixed[comp]]] = True
    free = ~fixed
    if not free.any():
        return V.copy()

    idx = np.flatnonzero(free)
    _, comp = connected_components(A[free][:, free], directed=False)
    A_ff = A[free][:, fixed].tocsr()
    fixed_idx = np.flatnonzero(fixed)
    boxes = []
    for k in range(comp.max() + 1):
        anchors = fixed_idx[np.unique(A_ff[comp == k].indices)]
        lo, hi = V[anchors].min(axis=0), V[anchors].max(axis=0)
        margin = _MARGIN * float(np.linalg.norm(hi - lo))
        boxes.append((lo - margin, hi + margin))

    X = V.copy()
    pending = np.ones(comp.max() + 1, dtype=bool)
    for o in range(degree, 0, -1):
        still = ~free
        still[idx[~pending[comp]]] = True  # the caps already solved do not move
        move = ~still
        M = L
        for _ in range(o - 1):
            M = L @ M
        M = M.tocsr()
        X[move] = spsolve(M[move][:, move].tocsc(), -(M[move][:, still] @ X[still]))
        if o == 1:
            break
        for k in np.flatnonzero(pending):
            lo, hi = boxes[k]
            P = X[idx[comp == k]]
            pending[k] = bool(np.any((P < lo) | (P > hi)))
        if not pending.any():
            break
    return X


def _remove_valence_3(F: np.ndarray, is_input: np.ndarray, fixed: np.ndarray
                      ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Removes the free vertices of valence 3 and leaves the triangle of their
    neighbors. When fairing, their three faces fold over one another.

    The removed vertex is marked fixed so that the system does not become singular.
    """
    tri_mesh = Mesh(np.zeros((F.max() + 1, 3)), F)
    valence = np.bincount(tri_mesh.edges.ravel(), minlength=len(fixed))
    candidates = np.flatnonzero((valence == 3) & ~fixed)
    if not len(candidates):
        return F, is_input, fixed

    existing = {tuple(t) for t in np.sort(F, axis=1).tolist()}
    incident: dict[int, list[int]] = {}
    for f, face in enumerate(F.tolist()):
        for v in face:
            incident.setdefault(v, []).append(f)

    F = F.copy()
    fixed = fixed.copy()
    delete = np.zeros(len(F), dtype=bool)
    for v in candidates.tolist():
        faces = incident.get(v, [])
        if len(faces) != 3 or delete[faces].any() or is_input[faces].any():
            continue
        # Each face, starting at v, contributes the edge a -> b of the ring.
        successor = {}
        for f in faces:
            t = F[f].tolist()
            k = t.index(v)
            successor[t[(k + 1) % 3]] = t[(k + 2) % 3]
        a = next(iter(successor))
        b = successor.get(a)
        c = successor.get(b)
        if c is None or successor.get(c) != a:
            continue
        new_face = tuple(sorted((a, b, c)))
        if new_face in existing:
            continue
        existing.add(new_face)
        F[faces[0]] = [a, b, c]
        delete[faces[1:]] = True
        fixed[v] = True
    return F[~delete], is_input[~delete], fixed


def _collapse_needles(V: np.ndarray, F: np.ndarray, is_input: np.ndarray,
                      fixed: np.ndarray, min_angle: float = 10.0
                      ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collapses the shortest edge of the triangles of the caps with an angle
    smaller than ``min_angle`` degrees.

    Only if the endpoints share exactly two neighbors (link condition),
    so that the surface remains a manifold.
    """
    tri_mesh = Mesh(V, F)
    needles = np.flatnonzero(~is_input & (np.degrees(tri_mesh.corner_angles.min(axis=1)) < min_angle))
    if not len(needles):
        return F, is_input, fixed

    neighbors: dict[int, set[int]] = {}
    for a, b in tri_mesh.edges.tolist():
        neighbors.setdefault(a, set()).add(b)
        neighbors.setdefault(b, set()).add(a)
    incident: dict[int, list[int]] = {}
    for f, face in enumerate(F.tolist()):
        for v in face:
            incident.setdefault(v, []).append(f)

    F = F.copy()
    fixed = fixed.copy()
    delete = np.zeros(len(F), dtype=bool)
    touched: set[int] = set()
    for f in needles.tolist():
        if delete[f]:
            continue
        t = F[f]
        lengths = [np.linalg.norm(V[t[(k + 2) % 3]] - V[t[(k + 1) % 3]]) for k in range(3)]
        k = int(np.argmin(lengths))
        a, b = int(t[(k + 1) % 3]), int(t[(k + 2) % 3])
        if fixed[a] and fixed[b]:
            continue
        if fixed[a]:
            a, b = b, a  # a disappears, so it must be free
        if a in touched or b in touched or len(neighbors[a] & neighbors[b]) != 2:
            continue
        faces_a = [g for g in incident[a] if not delete[g]]
        if is_input[faces_a].any():
            continue
        for g in faces_a:
            if b in F[g]:
                delete[g] = True
            else:
                F[g][F[g] == a] = b
        fixed[a] = True
        touched |= {a, b} | neighbors[a]
    return F[~delete], is_input[~delete], fixed


def _angle(p: np.ndarray, q: np.ndarray, r: np.ndarray) -> float:
    """Angle at ``p`` of the triangle ``pqr``."""
    u, v = q - p, r - p
    return float(np.arctan2(np.linalg.norm(np.cross(u, v)), u @ v))


def _flips(V: np.ndarray, F: np.ndarray, is_input: np.ndarray,
           max_passes: int = 50) -> np.ndarray:
    """Flips the edges of the caps that are not locally Delaunay.

    The flip is not done if the new edge already exists or if some new triangle
    ends up inverted.
    """
    F = F.copy()
    for _ in range(max_passes):
        tri_mesh = Mesh(V, F)
        existing = {tuple(a) for a in tri_mesh.edges.tolist()}
        # The two faces of each edge.
        fe = tri_mesh.face_edges.ravel()
        perm = np.argsort(fe, kind="stable")
        face = perm // 3
        start = np.searchsorted(fe[perm], np.arange(tri_mesh.n_edges))
        cand_edges = np.flatnonzero(tri_mesh.edge_face_count == 2)
        f1, f2 = face[start[cand_edges]], face[start[cand_edges] + 1]
        usable = ~is_input[f1] & ~is_input[f2]

        touched_faces: set[int] = set()
        done = 0
        for edge, g1, g2 in zip(cand_edges[usable], f1[usable], f2[usable]):
            if g1 in touched_faces or g2 in touched_faces:
                continue
            u, v = (int(x) for x in tri_mesh.edges[edge])
            t = list(F[g1])
            k = t.index(u)
            t = t[k:] + t[:k]
            if t[1] != v:  # g1 must traverse u -> v
                g1, g2 = g2, g1
                t = list(F[g1])
                k = t.index(u)
                t = t[k:] + t[:k]
            a, b, c = t
            d = next(x for x in F[g2] if x != a and x != b)
            if (min(c, d), max(c, d)) in existing:
                continue
            if _angle(V[c], V[a], V[b]) + _angle(V[d], V[a], V[b]) <= np.pi:
                continue
            old = np.cross(V[b] - V[a], V[c] - V[a]) + np.cross(V[a] - V[b], V[d] - V[b])
            n1 = np.cross(V[d] - V[a], V[c] - V[a])
            n2 = np.cross(V[b] - V[d], V[c] - V[d])
            if n1 @ old <= 0 or n2 @ old <= 0:
                continue
            F[g1] = [a, d, c]
            F[g2] = [d, b, c]
            existing.discard((min(a, b), max(a, b)))
            existing.add((min(c, d), max(c, d)))
            touched_faces |= {int(g1), int(g2)}
            done += 1
        if not done:
            break
    return F
