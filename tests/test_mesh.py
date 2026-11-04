"""Connectivity and basic geometry of Mesh."""

import numpy as np
import pytest

from gsd import Mesh, cube, icosphere, load_obj, punch_hole, save_obj, tetrahedron, torus

CLOSED_SHAPES = [
    ("tetrahedron", tetrahedron()),
    ("cube", cube()),
    ("icosphere", icosphere(2)),
    ("torus", torus(16, 8)),
]


@pytest.mark.parametrize("name,mesh", CLOSED_SHAPES)
def test_euler_and_closedness(name, mesh):
    """Every edge touched by exactly two faces, and Euler-Poincaré holds."""
    assert mesh.is_closed, f"{name} should be closed"
    assert mesh.is_edge_manifold

    # V - E + F = 2 - 2g. The torus has genus 1, the rest genus 0.
    chi = mesh.n_vertices - mesh.n_edges + mesh.n_faces
    expected = 0 if name == "torus" else 2
    assert chi == expected, f"{name}: Euler characteristic {chi}, expected {expected}"


@pytest.mark.parametrize("name,mesh", CLOSED_SHAPES)
def test_each_edge_traversed_in_both_directions(name, mesh):
    """In a closed oriented mesh, the two faces of an edge traverse it
    in opposite directions."""
    total = np.zeros(mesh.n_edges, dtype=np.int64)
    np.add.at(total, mesh.face_edges.ravel(), mesh.face_edge_sign.ravel().astype(np.int64))
    assert np.all(total == 0), f"{name}: there are edges traversed twice in the same direction"


@pytest.mark.parametrize("name,mesh", CLOSED_SHAPES)
def test_angles_of_each_face_sum_to_pi(name, mesh):
    ang = mesh.corner_angles
    assert np.allclose(ang.sum(axis=1), np.pi)
    assert np.all(ang > 0)


def test_sphere_area_converges():
    """When refining, the area of the icosphere tends to that of the unit sphere."""
    errors = [
        abs(icosphere(k).total_area - 4 * np.pi) / (4 * np.pi) for k in (1, 2, 3)
    ]
    assert errors[0] > errors[1] > errors[2]
    assert errors[-1] < 0.01


def test_sphere_normals_point_outward():
    mesh = icosphere(2)
    centroids = mesh.V[mesh.F].mean(axis=1)
    outward = np.einsum("ij,ij->i", mesh.face_normals, centroids)
    assert np.all(outward > 0), "there are faces with the normal pointing inward"


def test_punching_a_hole_leaves_a_boundary():
    mesh = icosphere(2)
    holed = punch_hole(mesh, np.array([0.0, 0.0, 1.0]), 0.5)
    assert holed.n_faces < mesh.n_faces
    assert not holed.is_closed
    assert len(holed.boundary_edges) > 0


def test_invariance_under_similarities():
    """Angles do not change when rotating, translating and scaling; the area scales
    with the square."""
    mesh = icosphere(2)
    theta = 0.7
    R = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ]
    )
    moved = mesh.transformed(R=R, t=np.array([3.0, -1.0, 2.0]), s=2.0)

    assert np.allclose(mesh.corner_angles, moved.corner_angles)
    assert np.isclose(moved.total_area, mesh.total_area * 4.0)


def test_flipping_the_orientation_inverts_the_normals():
    mesh = icosphere(1)
    flipped = mesh.with_flipped_orientation()
    assert np.allclose(mesh.face_normals, -flipped.face_normals)
    assert np.allclose(mesh.face_areas, flipped.face_areas)


def test_obj_round_trip(tmp_path):
    mesh = icosphere(1)
    path = tmp_path / "sphere.obj"
    save_obj(mesh, path)
    reloaded = load_obj(path)

    assert np.allclose(mesh.V, reloaded.V)
    assert np.array_equal(mesh.F, reloaded.F)
    assert reloaded.n_edges == mesh.n_edges


def test_obj_with_negative_indices_and_polygons(tmp_path):
    path = tmp_path / "quad.obj"
    path.write_text(
        "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\n"
        "f -4 -3 -2 -1\n",  # a quadrilateral with negative indices
        encoding="utf-8",
    )
    mesh = load_obj(path)
    assert mesh.n_faces == 2  # fan-triangulated
    assert np.isclose(mesh.total_area, 1.0)


def test_non_manifold_mesh_is_detected():
    """Three triangles that share an edge."""
    V = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0, -1, 0]], dtype=float)
    F = np.array([[0, 1, 2], [0, 1, 3], [0, 1, 4]], dtype=np.int32)
    mesh = Mesh(V, F)
    assert not mesh.is_edge_manifold
    assert mesh.edge_face_count.max() == 3


def test_out_of_range_indices_fail():
    with pytest.raises(ValueError):
        Mesh(np.zeros((3, 3)), np.array([[0, 1, 5]]))


@pytest.mark.parametrize("fraction", [0.05, 0.3, 1.0])
def test_splitting_edges_leaves_no_hanging_vertices(fraction):
    """With faces with one, two and three split edges, the mesh stays closed,
    with the same area and volume, and the old vertices do not move."""
    sphere = icosphere(2)
    to_split = np.random.default_rng(1).random(sphere.n_edges) < fraction
    fine = sphere.split_edges(to_split)

    assert fine.is_closed
    assert fine.n_vertices == sphere.n_vertices + to_split.sum()
    assert np.array_equal(fine.V[: sphere.n_vertices], sphere.V)
    assert fine.total_area == pytest.approx(sphere.total_area, rel=1e-12)

    def volume(m):
        a, b, c = m.V[m.F[:, 0]], m.V[m.F[:, 1]], m.V[m.F[:, 2]]
        return np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0

    assert volume(fine) == pytest.approx(volume(sphere), rel=1e-12)
