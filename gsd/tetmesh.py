"""Tetrahedral mesh. Face ``i`` of a tetrahedron is the one opposite its
vertex ``i``."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_EPS_VOL = 1e-14


@dataclass
class TetMesh:
    V: np.ndarray  # (n, 3) positions
    T: np.ndarray  # (m, 4) vertex indices

    def __post_init__(self) -> None:
        self.V = np.ascontiguousarray(self.V, dtype=np.float64)
        self.T = np.ascontiguousarray(self.T, dtype=np.int32)
        if self.V.ndim != 2 or self.V.shape[1] != 3:
            raise ValueError(f"V must be (n, 3), got {self.V.shape}")
        if self.T.ndim != 2 or self.T.shape[1] != 4:
            raise ValueError(f"T must be (m, 4), got {self.T.shape}")
        if self.n_tets and (self.T.min() < 0 or self.T.max() >= self.n_vertices):
            raise ValueError("T has out-of-range indices")
        self._orient()
        self._build_faces()

    @property
    def n_vertices(self) -> int:
        return self.V.shape[0]

    @property
    def n_tets(self) -> int:
        return self.T.shape[0]

    @property
    def n_faces(self) -> int:
        return self.faces.shape[0]

    def _signed_volumes(self) -> np.ndarray:
        p = self.V[self.T]
        return np.linalg.det(p[:, 1:] - p[:, :1]) / 6.0

    def _orient(self) -> None:
        """Flips the tetrahedra with negative volume."""
        negative = self._signed_volumes() < 0
        if negative.any():
            self.T[negative] = self.T[negative][:, [0, 2, 1, 3]]
        if np.abs(self._signed_volumes()).min() < _EPS_VOL:
            raise ValueError("there are degenerate tetrahedra (zero volume)")

    def _build_faces(self) -> None:
        T = self.T
        all_faces = np.stack([np.stack([T[:, (i + 1) % 4], T[:, (i + 2) % 4], T[:, (i + 3) % 4]], axis=1)
                          for i in range(4)], axis=1).reshape(-1, 3)
        self.faces, inv = np.unique(np.sort(all_faces, axis=1), axis=0, return_inverse=True)
        self.tet_faces = inv.reshape(-1, 4).astype(np.int32)
        # 1 = face on the boundary of the domain, 2 = interior.
        self.face_tet_count = np.bincount(self.tet_faces.ravel(),
                                          minlength=self.n_faces).astype(np.int32)

    @property
    def face_areas(self) -> np.ndarray:
        f = self.V[self.faces]
        return 0.5 * np.linalg.norm(np.cross(f[:, 1] - f[:, 0], f[:, 2] - f[:, 0]), axis=1)

    @property
    def tet_centroids(self) -> np.ndarray:
        return self.V[self.T].mean(axis=1)


def drop_degenerate_tets(V: np.ndarray, T: np.ndarray,
                         tol: float = 1e-12) -> tuple[np.ndarray, np.ndarray]:
    """Removes the tetrahedra with nearly zero volume (TetGen and Delaunay sometimes
    leave them)."""
    V = np.asarray(V, dtype=np.float64)
    T = np.asarray(T, dtype=np.int32)
    p = V[T]
    volume = np.abs(np.linalg.det(p[:, 1:] - p[:, :1])) / 6.0
    return V, T[volume > tol]


def tet_grid(n: int = 8, size: float = 2.0, center: np.ndarray | None = None) -> TetMesh:
    """Cube of side ``size`` split into ``n³`` small cubes, and each small cube into
    six tetrahedra (Kuhn subdivision)."""
    axes = np.linspace(-size / 2, size / 2, n + 1)
    X, Y, Z = np.meshgrid(axes, axes, axes, indexing="ij")
    V = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)

    # Each permutation of the axes is a path from (0,0,0) to (1,1,1) along edges
    # of the cube; its four vertices form one of the six tetrahedra.
    paths = []
    for perm in [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]:
        c = np.zeros(3, dtype=np.int64)
        path = [c.copy()]
        for ax in perm:
            c[ax] += 1
            path.append(c.copy())
        paths.append(path)
    paths = np.array(paths)  # (6, 4, 3)

    base = np.stack(np.meshgrid(np.arange(n), np.arange(n), np.arange(n), indexing="ij"),
                    axis=-1).reshape(-1, 1, 1, 3)
    corners = base + paths  # (n³, 6, 4, 3)
    T = ((corners[..., 0] * (n + 1) + corners[..., 1]) * (n + 1) + corners[..., 2]).reshape(-1, 4)

    c = np.zeros(3) if center is None else np.asarray(center, dtype=np.float64)
    return TetMesh(V + c, T)
