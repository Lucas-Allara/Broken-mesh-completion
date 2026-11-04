"""Fairing of the caps: the input does not move and the caps get closer
to the sphere they came from."""

import numpy as np
import pytest

from gsd.fairing import fair_filled_patches
from gsd.mesh import Mesh
from gsd.segmentation import graphcut_segment
from gsd.shapes import icosphere, random_holes

pytest.importorskip("tetgen", reason="the initial segmentation needs TetGen")


def _broken_sphere(subdivisions: int) -> Mesh:
    """Sphere with holes, without the loose vertices left by ``random_holes``."""
    broken = random_holes(icosphere(subdivisions), target_fraction=0.15, n_holes=4, seed=7)
    used, F = np.unique(broken.F, return_inverse=True)
    return Mesh(broken.V[used], F.reshape(-1, 3))


def _deviation_from_sphere(V: np.ndarray, F: np.ndarray) -> float:
    """Mean distance to the unit sphere, weighted by area."""
    T = V[F]
    area = 0.5 * np.linalg.norm(np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]), axis=1)
    P = np.concatenate([T.mean(axis=1), 0.5 * (T[:, 0] + T[:, 1]),
                        0.5 * (T[:, 1] + T[:, 2]), 0.5 * (T[:, 2] + T[:, 0])])
    weight = np.tile(area, 4)
    return float((np.abs(np.linalg.norm(P, axis=1) - 1.0) * weight).sum() / weight.sum())


@pytest.fixture(scope="module")
def case():
    broken = _broken_sphere(3)
    result = graphcut_segment(broken, gamma=3.0)
    V, F = fair_filled_patches(result["V"], result["F"], broken)
    return broken, result, V, F


def test_the_input_does_not_move(case):
    """The input vertices and faces that were in the output stay the same."""
    broken, result, V, F = case
    input_vertices = {tuple(p) for p in broken.V.tolist()}
    before = {tuple(p) for p in result["V"].tolist()} & input_vertices
    after = {tuple(p) for p in V.tolist()} & input_vertices
    assert before == after

    input_face_set = {frozenset(map(tuple, broken.V[f].tolist())) for f in broken.F}
    def present(V, F):
        return {c for c in (frozenset(map(tuple, V[f].tolist())) for f in F)
                if c in input_face_set}
    assert present(V, F) == present(result["V"], result["F"])


def test_fairing_does_not_open_the_surface(case):
    """No new boundaries or non-manifold edges appear."""
    _, result, V, F = case
    before = Mesh(result["V"], result["F"]).edge_face_count
    after = Mesh(V, F).edge_face_count
    assert np.all(after % 2 == 0)
    assert (after > 2).sum() <= (before > 2).sum()


def test_the_caps_get_closer_to_the_true_surface(case):
    _, result, V, F = case
    before = _deviation_from_sphere(result["V"], result["F"])
    after = _deviation_from_sphere(V, F)
    assert after < 0.5 * before


def test_without_caps_nothing_changes():
    intact = icosphere(2)
    result = graphcut_segment(intact, gamma=3.0)
    V, F = fair_filled_patches(result["V"], result["F"], intact)
    assert np.array_equal(V, result["V"])
    assert np.array_equal(F, result["F"])


def test_the_cap_does_not_fold():
    """On a refined tetrahedralization, where the method leaves no folds,
    no pair of neighboring faces is bent more than 120°."""
    broken = _broken_sphere(3)
    result = graphcut_segment(broken, gamma=3.0, max_far_edge=0.35)
    V, F = fair_filled_patches(result["V"], result["F"], broken, refine=1)
    tri_mesh = Mesh(V, F)
    fe = tri_mesh.face_edges.ravel()
    order = np.argsort(fe, kind="stable")
    face = order // 3
    start = np.searchsorted(fe[order], np.flatnonzero(tri_mesh.edge_face_count == 2))
    n = tri_mesh.face_normals
    cosine = np.einsum("ij,ij->i", n[face[start]], n[face[start + 1]])
    assert (cosine < np.cos(np.radians(120))).sum() == 0


def test_fairing_leaves_no_spikes():
    """On Homer with holes, nothing sticks out of the box of the real model by
    more than a little."""
    from pathlib import Path

    from gsd.conforming import conforming_tet_mesh
    from gsd.mesh import load_obj

    path = Path(__file__).resolve().parent.parent / "assets" / "homer.obj"
    if not path.exists():
        pytest.skip("assets/homer.obj is missing")
    homer = load_obj(path)
    lo, hi = homer.V.min(axis=0), homer.V.max(axis=0)
    truth = Mesh((homer.V - (lo + hi) / 2) * (1.35 / float((hi - lo).max())), homer.F)

    broken = random_holes(truth, target_fraction=0.22, n_holes=5, seed=3)
    used, F = np.unique(broken.F, return_inverse=True)
    broken = Mesh(broken.V[used], F.reshape(-1, 3))
    mesh, _ = conforming_tet_mesh(broken)
    result = graphcut_segment(broken, mesh=mesh, gamma=40.0, quadrature=1)
    V, _ = fair_filled_patches(result["V"], result["F"], broken)

    lo, hi = truth.V.min(axis=0), truth.V.max(axis=0)
    overshoot = np.maximum(V - hi, lo - V).max()
    assert overshoot < 0.03
