"""Symmetry completion. The test shapes are deformed spheres,
symmetric with respect to x = 0."""

import numpy as np
import pytest
from scipy.spatial import cKDTree

from gsd.mesh import Mesh
from gsd.shapes import icosphere, punch_hole
from gsd.symmetry import _samples, mirror_completion, reconstruct_with_symmetry, symmetry_plane


def _with_bumps() -> Mesh:
    """Stretched sphere with two bumps, symmetric only with respect to x = 0."""
    sphere = icosphere(3)
    V = sphere.V * np.array([1.0, 0.8, 1.2])
    for side in (-1.0, 1.0):
        center = np.array([side * 0.55, 0.3, 0.75])
        V = V + 0.35 * np.exp(-np.sum((V - center) ** 2, axis=1) / 0.04)[:, None] * V
    return Mesh(V, sphere.F)


def _rotation(seed: int) -> np.ndarray:
    q, r = np.linalg.qr(np.random.default_rng(seed).standard_normal((3, 3)))
    return q * np.sign(np.diag(r))


def test_the_plane_is_found_even_if_the_shape_is_rotated():
    shape = _with_bumps()
    R, t = _rotation(4), np.array([0.3, -0.2, 0.1])
    rotated = Mesh(shape.V @ R.T + t, shape.F)

    n, d, residual = symmetry_plane(rotated)
    true_normal = R @ np.array([1.0, 0.0, 0.0])
    angle = np.degrees(np.arccos(min(1.0, abs(n @ true_normal))))
    assert angle < 1.0
    assert abs(abs(d) - abs(true_normal @ t)) < 1e-2
    assert residual < 0.5 * np.median(rotated.edge_lengths)


def test_a_model_without_symmetry_is_recognized():
    """Lucy is not symmetric: the residual exceeds half an edge.

    A deformed sphere does not work for this: discarding the 20% that matches worst,
    it always finds a plane that nearly works."""
    from pathlib import Path

    from gsd.mesh import load_obj

    path = Path(__file__).resolve().parent.parent / "assets" / "lucy.obj"
    if not path.exists():
        pytest.skip("assets/lucy.obj is missing")
    lucy = load_obj(path)
    _, _, residual = symmetry_plane(lucy)
    assert residual > 0.5 * np.median(lucy.edge_lengths)


def _with_horns() -> Mesh:
    """Like ``_with_bumps`` but finer and with more pronounced protrusions."""
    sphere = icosphere(4)
    V = sphere.V * np.array([1.0, 0.8, 1.2])
    for side in (-1.0, 1.0):
        center = np.array([side * 0.55, 0.3, 0.75])
        V = V + 0.6 * np.exp(-np.sum((V - center) ** 2, axis=1) / 0.03)[:, None] * V
    return Mesh(V, sphere.F)


def test_what_is_missing_on_one_side_comes_back_from_the_other():
    """Without the tip of a protrusion, mirroring the other one leaves the reconstruction
    closer to the original shape than fairing alone."""
    pytest.importorskip("tetgen")
    shape = _with_horns()
    tip = shape.V[np.argmax(shape.V @ np.array([0.55, 0.3, 0.75]))]
    broken = punch_hole(shape, tip, 0.4)
    used, F = np.unique(broken.F, return_inverse=True)
    broken = Mesh(broken.V[used], F.reshape(-1, 3))

    result = reconstruct_with_symmetry(broken)
    assert result["symmetry"] is not None
    assert result["mirrored"].n_faces > 0

    # The faces of the shape that are missing from the input.
    input_set = {tuple(sorted(map(tuple, p.tolist()))) for p in broken.V[broken.F]}
    missing = np.array([tuple(sorted(map(tuple, shape.V[t].tolist()))) not in input_set
                        for t in shape.F])
    rng = np.random.default_rng(0)
    truth = _samples(shape.V, shape.F[missing], 20_000, rng)

    def error(VF):
        return cKDTree(_samples(VF[0], VF[1], 200_000, rng)).query(truth)[0].mean()

    assert error(result["symmetry"]) < 0.6 * error(result["faired"])


def test_what_is_already_there_is_not_mirrored():
    """Without holes nothing is mirrored."""
    shape = _with_bumps()
    n, d, _ = symmetry_plane(shape)
    assert mirror_completion(shape, shape, n, d).n_faces == 0
