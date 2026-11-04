"""Triangle mesh with its connectivity.

Convention: corner ``i`` of a face is opposite the edge
``F[f, i+1] -> F[f, i+2]``. Each edge is stored once as ``(min, max)`` and
``face_edge_sign[f, i]`` is +1 if the face traverses it in that direction and -1
otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Avoids dividing by zero when normalizing the normals of degenerate triangles.
_EPS_AREA = 1e-14


@dataclass
class Mesh:
    V: np.ndarray  # (n, 3) positions
    F: np.ndarray  # (m, 3) vertex indices

    def __post_init__(self) -> None:
        self.V = np.ascontiguousarray(self.V, dtype=np.float64)
        self.F = np.ascontiguousarray(self.F, dtype=np.int32)
        if self.V.ndim != 2 or self.V.shape[1] != 3:
            raise ValueError(f"V must be (n, 3), got {self.V.shape}")
        if self.F.ndim != 2 or self.F.shape[1] != 3:
            raise ValueError(f"F must be (m, 3), got {self.F.shape}")
        if self.n_faces and (self.F.min() < 0 or self.F.max() >= self.n_vertices):
            raise ValueError("F has out-of-range indices")
        self._build_edges()

    @property
    def n_vertices(self) -> int:
        return self.V.shape[0]

    @property
    def n_faces(self) -> int:
        return self.F.shape[0]

    @property
    def n_edges(self) -> int:
        return self.edges.shape[0]

    def _build_edges(self) -> None:
        F = self.F
        directed = np.stack(
            [np.stack([F[:, (i + 1) % 3], F[:, (i + 2) % 3]], axis=1) for i in range(3)],
            axis=1,
        ).reshape(-1, 2)

        self.edges, inv = np.unique(np.sort(directed, axis=1), axis=0, return_inverse=True)
        self.face_edges = inv.reshape(-1, 3).astype(np.int32)
        self.face_edge_sign = np.where(directed[:, 0] < directed[:, 1], 1, -1
                                       ).reshape(-1, 3).astype(np.int8)
        # 1 = boundary, 2 = interior, more than 2 = non-manifold.
        self.edge_face_count = np.bincount(self.face_edges.ravel(),
                                           minlength=self.n_edges).astype(np.int32)

    @property
    def boundary_edges(self) -> np.ndarray:
        return np.flatnonzero(self.edge_face_count == 1)

    @property
    def is_edge_manifold(self) -> bool:
        return bool(np.all(self.edge_face_count <= 2))

    @property
    def is_closed(self) -> bool:
        return bool(np.all(self.edge_face_count == 2))

    @property
    def face_corners(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.V[self.F[:, 0]], self.V[self.F[:, 1]], self.V[self.F[:, 2]]

    @property
    def face_normals_raw(self) -> np.ndarray:
        """Unnormalized normals (their norm is twice the area)."""
        a, b, c = self.face_corners
        return np.cross(b - a, c - a)

    @property
    def face_areas(self) -> np.ndarray:
        return 0.5 * np.linalg.norm(self.face_normals_raw, axis=1)

    @property
    def face_normals(self) -> np.ndarray:
        n = self.face_normals_raw
        return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), _EPS_AREA)

    @property
    def total_area(self) -> float:
        return float(self.face_areas.sum())

    @property
    def corner_angles(self) -> np.ndarray:
        """(m, 3) interior angle of each corner."""
        p = self.V[self.F]
        angles = np.empty((self.n_faces, 3))
        for i in range(3):
            u = p[:, (i + 1) % 3] - p[:, i]
            v = p[:, (i + 2) % 3] - p[:, i]
            # atan2 instead of arccos: it keeps its precision near 0 and pi.
            angles[:, i] = np.arctan2(np.linalg.norm(np.cross(u, v), axis=1),
                                      np.einsum("ij,ij->i", u, v))
        return angles

    @property
    def edge_lengths(self) -> np.ndarray:
        e = self.edges
        return np.linalg.norm(self.V[e[:, 1]] - self.V[e[:, 0]], axis=1)

    @property
    def mean_edge_length(self) -> float:
        return float(self.edge_lengths.mean())

    def bounding_box(self) -> tuple[np.ndarray, np.ndarray]:
        return self.V.min(axis=0), self.V.max(axis=0)

    def transformed(self, R: np.ndarray | None = None, t: np.ndarray | None = None,
                    s: float = 1.0) -> Mesh:
        """Scaled, rotated and translated copy."""
        V = self.V * s
        if R is not None:
            V = V @ np.asarray(R, dtype=np.float64).T
        if t is not None:
            V = V + np.asarray(t, dtype=np.float64)
        return Mesh(V, self.F.copy())

    def with_flipped_orientation(self) -> Mesh:
        return Mesh(self.V.copy(), self.F[:, ::-1].copy())

    def without_faces(self, drop: np.ndarray) -> Mesh:
        keep = np.ones(self.n_faces, dtype=bool)
        keep[np.asarray(drop, dtype=np.int64)] = False
        return Mesh(self.V.copy(), self.F[keep].copy())

    def split_edges(self, to_split: np.ndarray) -> Mesh:
        """Splits at their midpoint the edges marked in ``to_split`` (a mask over
        ``edges``) and retriangulates each face according to how many of its edges
        were split.

        The old vertices keep their index and the midpoints go at the end.
        """
        to_split = np.asarray(to_split, dtype=bool)
        idx = np.flatnonzero(to_split)
        mid = np.full(self.n_edges, -1, dtype=np.int64)
        mid[idx] = self.n_vertices + np.arange(len(idx))
        V = np.vstack([self.V, 0.5 * (self.V[self.edges[idx, 0]] + self.V[self.edges[idx, 1]])])

        F = self.F.astype(np.int64)
        m = mid[self.face_edges]  # m[:, k]: midpoint of the edge opposite k
        s = to_split[self.face_edges]
        n = s.sum(axis=1)
        faces = [F[n == 0]]

        # With one or two split edges the face is rotated so that corner 0
        # is opposite the distinct edge (the split one or the unsplit one).
        for count, distinct in ((1, True), (2, False)):
            sel = n == count
            k = np.argmax(s[sel] == distinct, axis=1)
            row = np.arange(sel.sum())
            p0, p1, p2 = (F[sel][row, (k + j) % 3] for j in range(3))
            m0, m1, m2 = (m[sel][row, (k + j) % 3] for j in range(3))
            if count == 1:
                new = [(p0, p1, m0), (p0, m0, p2)]
            else:
                new = [(p0, m2, m1), (m2, p1, p2), (m2, p2, m1)]
            faces += [np.column_stack(t) for t in new]

        sel = n == 3
        a, b, c = F[sel].T
        ma, mb, mc = m[sel].T
        faces += [np.column_stack(t) for t in
                  ((a, mc, mb), (mc, b, ma), (mb, ma, c), (ma, mb, mc))]
        return Mesh(V, np.vstack(faces))


def load_obj(path: str | Path) -> Mesh:
    """Reads the vertices and faces of an OBJ. Polygons are fan-triangulated."""
    vertices: list[list[float]] = []
    faces: list[tuple[int, int, int]] = []

    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith("v "):
                parts = line.split()
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith("f "):
                # The tokens can be "v", "v/vt", "v//vn" or "v/vt/vn".
                idx = [int(tok.split("/")[0]) for tok in line.split()[1:]]
                # OBJ counts from 1; negative indices count from the end.
                idx = [i - 1 if i > 0 else len(vertices) + i for i in idx]
                for k in range(1, len(idx) - 1):
                    faces.append((idx[0], idx[k], idx[k + 1]))

    if not faces:
        raise ValueError(f"{path} has no faces")
    return Mesh(np.array(vertices), np.array(faces))


def save_obj(mesh: Mesh, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for v in mesh.V:
            fh.write(f"v {v[0]:.17g} {v[1]:.17g} {v[2]:.17g}\n")
        for f in mesh.F + 1:
            fh.write(f"f {f[0]} {f[1]} {f[2]}\n")
