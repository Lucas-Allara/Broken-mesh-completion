"""Optional constraints of Section 5.2 of the paper.

They cannot be imposed inside the graph cut, so the paper applies them with
two heuristics on the labeling given by the cut:

- Every input face must be a face of some inside tetrahedron. If it is not,
  its neighboring tetrahedron with the largest w is moved inside.
- The output surface must be a manifold. At each edge or vertex where it is not,
  the tetrahedra that touch it are moved inside, from largest to smallest
  w, until it is.

After each step the energy of Equation 8 is improved by flipping neighbors
of what changed, without breaking constraints that already held.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from .mesh import Mesh
from .tetmesh import TetMesh


def input_faces(surf: Mesh, mesh: TetMesh) -> np.ndarray:
    """Indices in ``mesh.faces`` of the triangles of ``surf`` that are faces of
    the tetrahedralization (with exactly equal vertices)."""
    from scipy.spatial import cKDTree

    d, ids = cKDTree(mesh.V).query(np.asarray(surf.V))
    diagonal = float(np.linalg.norm(mesh.V.max(axis=0) - mesh.V.min(axis=0)))
    F = np.asarray(surf.F)
    F = F[(d <= 1e-12 * diagonal)[F].all(axis=1)]
    index = {tuple(c): k for k, c in enumerate(np.sort(mesh.faces, axis=1).tolist())}
    face_ids = (index.get(tuple(sorted(int(ids[v]) for v in t))) for t in F)
    return np.array([c for c in face_ids if c is not None], dtype=np.int64)


def _csr(keys: np.ndarray, values: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(keys, kind="stable")
    return np.searchsorted(keys[order], np.arange(n + 1)), values[order]


class _Labeling:
    """The label of each tetrahedron and what is needed to change it one at a time."""

    def __init__(self, mesh: TetMesh, w, inside, u_inside, u_outside, tet_a, tet_b, v_weights,
                 inputs: np.ndarray) -> None:
        m, nf = mesh.n_tets, mesh.n_faces
        self.w = np.asarray(w)
        self.x = np.asarray(inside, dtype=bool).copy()
        self.cost = (np.asarray(u_outside), np.asarray(u_inside))
        self.T = mesh.T
        self.tet_face_ids = mesh.tet_faces.astype(np.int64)
        self.manifold = False

        tf = self.tet_face_ids.ravel()
        tet = np.repeat(np.arange(m), 4)
        order = np.argsort(tf, kind="stable")
        tf_o, tet_o = tf[order], tet[order]
        first = np.ones(len(tf_o), dtype=bool)
        first[1:] = tf_o[1:] != tf_o[:-1]
        self.face_tets = np.full((nf, 2), -1, dtype=np.int64)
        self.face_tets[tf_o[first], 0] = tet_o[first]
        self.face_tets[tf_o[~first], 1] = tet_o[~first]
        # Inside tetrahedra of each face: the face belongs to the output if there is exactly one.
        self.n_inside = np.bincount(tf[self.x[tet]], minlength=nf)

        self.is_input = np.zeros(nf, dtype=bool)
        self.is_input[inputs] = True

        face_ids = mesh.faces.astype(np.int64)
        edges = np.sort(np.concatenate([face_ids[:, [0, 1]], face_ids[:, [1, 2]], face_ids[:, [2, 0]]]), axis=1)
        self.edges, inv = np.unique(edges, axis=0, return_inverse=True)
        self.face_edges = inv.reshape(3, nf).T
        ne = len(self.edges)
        self.edge_faces = _csr(self.face_edges.ravel(), np.repeat(np.arange(nf), 3), ne)
        self.vertex_faces = _csr(face_ids.ravel(), np.repeat(np.arange(nf), 3), mesh.n_vertices)
        self.vertex_tets = _csr(self.T.ravel().astype(np.int64), tet, mesh.n_vertices)
        self.tet_edges = np.sort(self.face_edges[self.tet_face_ids].reshape(m, 12), axis=1)[:, ::2]
        self.neighbors = _csr(np.concatenate([tet_a, tet_b]),
                              np.stack([np.concatenate([tet_b, tet_a]),
                                        np.arange(2 * len(tet_a)) % len(tet_a)], axis=1), m)
        self.v_weights = np.asarray(v_weights)

    @staticmethod
    def _of(csr, k):
        starts, values = csr
        return values[starts[k]:starts[k + 1]]

    def flip(self, i: int) -> None:
        self.x[i] = not self.x[i]
        self.n_inside[self.tet_face_ids[i]] += 1 if self.x[i] else -1

    def edge_ok(self, e: int) -> bool:
        return int((self.n_inside[self._of(self.edge_faces, e)] == 1).sum()) in (0, 2)

    def vertex_ok(self, v: int) -> bool:
        face_ids = self._of(self.vertex_faces, v)
        face_ids = face_ids[self.n_inside[face_ids] == 1]
        if len(face_ids) == 0:
            return True
        # The output faces around v must form a single fan,
        # joined by the edges that contain v.
        parent = list(range(len(face_ids)))

        def root(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        first_face = {}
        for k, c in enumerate(face_ids):
            for e in self.face_edges[c]:
                if v in self.edges[e]:
                    if e in first_face:
                        parent[root(k)] = root(first_face[e])
                    else:
                        first_face[e] = k
        return len({root(k) for k in range(len(face_ids))}) == 1

    def delta(self, i: int) -> float:
        current = bool(self.x[i])
        nb = self._of(self.neighbors, i)
        j, weight = nb[:, 0], self.v_weights[nb[:, 1]]
        xj = self.x[j]
        return float(self.cost[not current][i] - self.cost[current][i]
                     + (weight * ((xj == current).astype(float) - (xj != current))).sum())

    def try_flip(self, i: int) -> bool:
        """Flips ``i`` if it does not break any constraint that held."""
        if self.x[i]:
            face_ids = self.tet_face_ids[i]
            if np.any(self.is_input[face_ids] & (self.n_inside[face_ids] == 1)):
                return False
        if not self.manifold:
            self.flip(i)
            return True
        edges, vertices = self.tet_edges[i], self.T[i]
        before = [self.edge_ok(e) for e in edges] + [self.vertex_ok(v) for v in vertices]
        self.flip(i)
        after = [self.edge_ok(e) for e in edges] + [self.vertex_ok(v) for v in vertices]
        if any(a and not d for a, d in zip(before, after)):
            self.flip(i)
            return False
        return True

    def improve(self, changed) -> None:
        """Lowers the energy by flipping neighbors of the changed tetrahedra."""
        queue = deque(int(j) for i in changed for j in self._of(self.neighbors, i)[:, 0])
        while queue:
            j = queue.popleft()
            if self.delta(j) < -1e-12 and self.try_flip(j):
                queue.extend(int(k) for k in self._of(self.neighbors, j)[:, 0])

    def move_inside(self, tets, satisfied) -> None:
        """Moves inside the outside tetrahedra of ``tets``, from largest to smallest w,
        until ``satisfied()``."""
        tets = tets[~self.x[tets]]
        changed = []
        for i in tets[np.argsort(-self.w[tets], kind="stable")]:
            if satisfied():
                break
            self.flip(int(i))
            changed.append(int(i))
        self.improve(changed)


def enforce_optional_constraints(mesh: TetMesh, inputs: np.ndarray, w, inside,
                                 u_inside, u_outside, tet_a, tet_b, v_weights,
                                 max_rounds: int = 50) -> np.ndarray:
    """Applies the two heuristics of Section 5.2 to the labeling ``inside``.

    ``inputs`` are the indices of the faces of ``mesh`` that come from the input
    (``input_faces``). Returns the new labeling.
    """
    e = _Labeling(mesh, w, inside, u_inside, u_outside, tet_a, tet_b, v_weights, inputs)

    for f in inputs:
        if e.n_inside[f] == 0:
            tets = e.face_tets[f][e.face_tets[f] >= 0]
            i = int(tets[np.argmax(e.w[tets])])
            e.flip(i)
            e.improve([i])

    e.manifold = True
    for _ in range(max_rounds):
        output = e.n_inside == 1
        count = np.bincount(e.face_edges[output].ravel(), minlength=len(e.edges))
        bad_edges = np.flatnonzero((count != 0) & (count != 2))
        for a in bad_edges:
            if not e.edge_ok(a):
                tets = np.unique(e.face_tets[e._of(e.edge_faces, a)])
                e.move_inside(tets[tets >= 0], lambda a=a: e.edge_ok(a))
        bad_vertices = [v for v in np.unique(mesh.faces[e.n_inside == 1]) if not e.vertex_ok(v)]
        for v in bad_vertices:
            if not e.vertex_ok(v):
                e.move_inside(e._of(e.vertex_tets, v), lambda v=v: e.vertex_ok(v))
        if len(bad_edges) == 0 and len(bad_vertices) == 0:
            break
    return e.x
