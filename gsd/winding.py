"""Generalized winding number (Jacobson, Kavan and Sorkine-Hornung, 2013).

``winding_number`` sums the solid angle of all the faces; ``WindingHierarchy``
is the hierarchical evaluation of Section 4.3, which gives the same result while
summing far fewer triangles.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .mesh import Mesh

_INV_4PI = 1.0 / (4.0 * np.pi)

# Elements per block when evaluating many points, to bound memory.
_CHUNK_ELEMS = 4_000_000


def solid_angles(V: np.ndarray, F: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Signed solid angle of each face seen from ``p`` (Van Oosterom and Strackee
    formula)."""
    a = V[F[:, 0]] - p
    b = V[F[:, 1]] - p
    c = V[F[:, 2]] - p
    la = np.linalg.norm(a, axis=-1)
    lb = np.linalg.norm(b, axis=-1)
    lc = np.linalg.norm(c, axis=-1)

    num = np.einsum("...i,...i->...", a, np.cross(b, c))
    den = (la * lb * lc
           + np.einsum("...i,...i->...", a, b) * lc
           + np.einsum("...i,...i->...", b, c) * la
           + np.einsum("...i,...i->...", c, a) * lb)
    return 2.0 * np.arctan2(num, den)


def winding_number(mesh: Mesh, points: np.ndarray) -> np.ndarray:
    P = np.atleast_2d(np.asarray(points, dtype=np.float64))
    if P.shape[1] != 3:
        raise ValueError(f"points must be (k, 3), got {P.shape}")
    return _direct_sum(mesh.V, mesh.F, P)


def _direct_sum(V: np.ndarray, F: np.ndarray, P: np.ndarray) -> np.ndarray:
    out = np.zeros(P.shape[0])
    if len(F) == 0:
        return out
    step = max(1, _CHUNK_ELEMS // len(F))
    for i in range(0, P.shape[0], step):
        block = P[i:i + step][:, None, :]
        out[i:i + step] = solid_angles(V, F, block).sum(axis=1) * _INV_4PI
    return out


def _closure(F: np.ndarray) -> np.ndarray:
    """Triangles that close the surface patch ``F``.

    Each edge that ``F`` traverses more times in one direction than in the other
    (its boundary) is joined to a fixed vertex with a triangle that traverses it
    backwards. Patch plus closure is closed, which is all that is needed.
    """
    d = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    sign = np.where(d[:, 0] < d[:, 1], 1, -1)
    keys, inv = np.unique(np.sort(d, axis=1), axis=0, return_inverse=True)
    count = np.rint(np.bincount(inv.ravel(), weights=sign,
                                minlength=len(keys))).astype(np.int64)
    boundary = count != 0
    i, j, c = keys[boundary, 0], keys[boundary, 1], count[boundary]

    k = int(F[0, 0])
    tri = np.where((c > 0)[:, None],
                   np.column_stack([j, i, np.full_like(i, k)]),
                   np.column_stack([i, j, np.full_like(i, k)]))
    tri = np.repeat(tri, np.abs(c), axis=0)
    # The ones that touch k are degenerate.
    return tri[(tri[:, 0] != k) & (tri[:, 1] != k)]


@dataclass
class _Node:
    lo: np.ndarray           # box of the vertices of its faces
    hi: np.ndarray
    closure: np.ndarray
    faces: np.ndarray | None  # only in the leaves
    children: tuple[_Node, _Node] | None


class WindingHierarchy:
    """Exact hierarchical evaluation (Algorithms 1 and 2 of the paper).

    If a point lies outside the box of a patch ``S``, ``w_S(p)`` is minus the
    winding number of any surface that closes it, and the closure has far fewer
    faces than ``S``. The tree splits the box in half along its longest side.
    """

    def __init__(self, mesh, leaf_size: int = 100) -> None:
        self.V = np.asarray(mesh.V, dtype=np.float64)
        self.leaf_size = leaf_size
        F = np.asarray(mesh.F, dtype=np.int64)
        self._eps = 1e-9 * float(np.linalg.norm(self.V.max(axis=0) - self.V.min(axis=0)))
        self.root = self._build(F, *self._box(F))

    def _box(self, F: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        P = self.V[F.ravel()]
        return P.min(axis=0), P.max(axis=0)

    def _build(self, F: np.ndarray, lo_part: np.ndarray, hi_part: np.ndarray) -> _Node:
        # The "far away" test uses the actual box of the faces, not that of the
        # partition, because a face can stick out of the box of its child.
        lo, hi = self._box(F)
        closure = _closure(F)
        if len(F) < self.leaf_size or len(closure) >= len(F):
            return _Node(lo, hi, closure, F, None)

        split_axis = int(np.argmax(hi_part - lo_part))
        cut = 0.5 * (lo_part[split_axis] + hi_part[split_axis])
        left = self.V[F].mean(axis=1)[:, split_axis] < cut
        if left.all() or not left.any():
            return _Node(lo, hi, closure, F, None)

        hi_left, lo_right = hi_part.copy(), lo_part.copy()
        hi_left[split_axis] = lo_right[split_axis] = cut
        children = (self._build(F[left], lo_part, hi_left),
                    self._build(F[~left], lo_right, hi_part))
        return _Node(lo, hi, closure, None, children)

    def __call__(self, points: np.ndarray) -> np.ndarray:
        P = np.atleast_2d(np.asarray(points, dtype=np.float64))
        if P.shape[1] != 3:
            raise ValueError(f"points must be (k, 3), got {P.shape}")
        out = np.zeros(P.shape[0])
        # All the points go down together: at each node the ones outside the
        # box are resolved with the closure and the rest move on to the children.
        pending = [(self.root, np.arange(P.shape[0]))]
        while pending:
            node, idx = pending.pop()
            Q = P[idx]
            if node.children is None:
                out[idx] += _direct_sum(self.V, node.faces, Q)
                continue
            outside = np.any((Q < node.lo - self._eps) | (Q > node.hi + self._eps), axis=1)
            if outside.any():
                out[idx[outside]] -= _direct_sum(self.V, node.closure, Q[outside])
            inside = idx[~outside]
            if len(inside):
                pending += [(h, inside) for h in node.children]
        return out


def sample_grid(mesh: Mesh, resolution: int = 64, padding: float = 0.25,
                axis: int = 2, offset: float = 0.0):
    """Evaluates w on a plane slice of the bounding box.

    Returns ``(w, extent)``, ready for ``matplotlib.imshow``.
    """
    lo, hi = mesh.bounding_box()
    margin = padding * (hi - lo).max()
    lo, hi = lo - margin, hi + margin

    axes = [i for i in range(3) if i != axis]
    u = np.linspace(lo[axes[0]], hi[axes[0]], resolution)
    v = np.linspace(lo[axes[1]], hi[axes[1]], resolution)
    U, W = np.meshgrid(u, v, indexing="xy")

    P = np.empty((U.size, 3))
    P[:, axes[0]] = U.ravel()
    P[:, axes[1]] = W.ravel()
    P[:, axis] = offset
    return winding_number(mesh, P).reshape(resolution, resolution), (u[0], u[-1], v[0], v[-1])
