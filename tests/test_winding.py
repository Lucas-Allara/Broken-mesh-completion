"""Generalized winding number, direct and hierarchical."""

import numpy as np
import pytest

from gsd import (
    Mesh,
    cube,
    icosphere,
    punch_hole,
    tetrahedron,
    torus,
    winding_number,
)

CLOSED_SHAPES = [
    ("tetrahedron", tetrahedron(), np.zeros(3)),
    ("cube", cube(), np.zeros(3)),
    ("icosphere", icosphere(2), np.zeros(3)),
    ("torus", torus(32, 16), np.array([1.0, 0.0, 0.0])),  # on the tube, not the hole
]


# --------------------------------------------------------------- closed mesh


@pytest.mark.parametrize("name,mesh,inside", CLOSED_SHAPES)
def test_equals_one_inside(name, mesh, inside):
    w = winding_number(mesh, inside[None, :])[0]
    assert np.isclose(w, 1.0, atol=1e-9), f"{name}: inside gave {w}"


@pytest.mark.parametrize("name,mesh,inside", CLOSED_SHAPES)
def test_equals_zero_outside(name, mesh, inside):
    far = np.array([[10.0, 7.0, -13.0]])
    w = winding_number(mesh, far)[0]
    assert np.isclose(w, 0.0, atol=1e-9), f"{name}: outside gave {w}"


def test_the_hole_of_the_torus_is_outside():
    """The origin is in the hole of the torus, outside the solid."""
    w = winding_number(torus(32, 16), np.zeros((1, 3)))[0]
    assert np.isclose(w, 0.0, atol=1e-9)


def test_unit_jump_when_crossing_the_surface():
    """When crossing the surface, w jumps by one unit."""
    mesh = icosphere(3)
    n = np.array([0.0, 0.0, 1.0])
    w_in = winding_number(mesh, (n * 0.995)[None, :])[0]
    w_out = winding_number(mesh, (n * 1.005)[None, :])[0]
    assert np.isclose(w_in - w_out, 1.0, atol=0.02)


def test_sharp_transition_on_an_intact_mesh():
    """On a closed mesh the transition happens in a single sample."""
    mesh = icosphere(3)
    t = np.linspace(0.0, 2.0, 200)
    P = np.stack([np.zeros_like(t), np.zeros_like(t), t], axis=1)
    w = winding_number(mesh, P)
    assert np.max(np.abs(np.diff(w))) > 0.8


# --------------------------------------------------------------- orientation


def test_flipping_the_mesh_inverts_the_sign():
    mesh = icosphere(2)
    P = np.array([[0.0, 0.0, 0.0], [0.3, 0.2, -0.1], [5.0, 5.0, 5.0]])
    assert np.allclose(
        winding_number(mesh, P), -winding_number(mesh.with_flipped_orientation(), P)
    )


# --------------------------------------------------------------- invariances


def test_invariant_under_rigid_motion():
    mesh = icosphere(2)
    P = np.array([[0.0, 0.0, 0.0], [0.5, 0.1, 0.2], [1.4, 0.0, 0.0]])

    theta = 0.9
    R = np.array(
        [[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]]
    )
    t = np.array([-2.0, 4.0, 1.5])

    assert np.allclose(winding_number(mesh, P), winding_number(mesh.transformed(R=R, t=t), P @ R.T + t))


def test_invariant_under_scaling():
    mesh = icosphere(2)
    P = np.array([[0.0, 0.0, 0.0], [0.5, 0.1, 0.2], [3.0, 0.0, 0.0]])
    assert np.allclose(winding_number(mesh, P), winding_number(mesh.transformed(s=7.0), P * 7.0))


def test_invariant_under_mesh_refinement():
    P = np.array([[0.0, 0.0, 0.0], [0.4, 0.3, 0.2], [2.0, 0.0, 0.0]])
    a = winding_number(icosphere(2), P)
    b = winding_number(icosphere(4), P)
    assert np.allclose(a, b, atol=1e-6)


# ------------------------------------------------------------- broken geometry


def test_punching_a_hole_lowers_the_inside_value():
    """Without a cap, w at the center drops by the fraction of the area that is missing."""
    mesh = icosphere(4)
    holed = punch_hole(mesh, np.array([0.0, 0.0, 1.0]), 0.5)

    w = winding_number(holed, np.zeros((1, 3)))[0]
    assert 0.85 < w < 0.99, f"value at the center: {w}"

    missing = 1.0 - holed.total_area / mesh.total_area
    assert np.isclose(1.0 - w, missing, atol=0.02)


def test_larger_hole_lowers_the_value_more():
    mesh = icosphere(4)
    values = [
        winding_number(punch_hole(mesh, np.array([0.0, 0.0, 1.0]), r), np.zeros((1, 3)))[0]
        for r in (0.3, 0.6, 0.9)
    ]
    assert values[0] > values[1] > values[2]


def test_the_transition_smooths_out_near_the_hole():
    """On a segment that goes through the hole, w changes gradually instead of
    jumping."""
    mesh = icosphere(3)
    holed = punch_hole(mesh, np.array([0.0, 0.0, 1.0]), 0.6)

    t = np.linspace(0.5, 1.5, 300)
    P = np.stack([np.zeros_like(t), np.zeros_like(t), t], axis=1)

    jump_intact = np.max(np.abs(np.diff(winding_number(mesh, P))))
    jump_broken = np.max(np.abs(np.diff(winding_number(holed, P))))

    assert jump_intact > 0.8, "the closed mesh should jump by one unit"
    assert jump_broken < 0.1, "the broken mesh should not jump"


def test_far_from_the_hole_it_still_classifies_well():
    """On the side opposite the hole, w stays close to 1."""
    holed = punch_hole(icosphere(4), np.array([0.0, 0.0, 1.0]), 0.5)
    w = winding_number(holed, np.array([[0.0, 0.0, -0.9]]))[0]
    assert w > 0.9


def test_decays_to_zero_far_away():
    holed = punch_hole(icosphere(3), np.array([0.0, 0.0, 1.0]), 0.6)
    far_points = np.array([[50.0, 0.0, 0.0], [0.0, 80.0, 0.0], [0.0, 0.0, -60.0]])
    assert np.allclose(winding_number(holed, far_points), 0.0, atol=1e-3)


# ------------------------------------------------------------------ mechanics


def test_block_query_gives_the_same():
    from gsd import winding

    mesh = icosphere(3)
    rng = np.random.default_rng(0)
    P = rng.normal(size=(500, 3))

    expected = winding_number(mesh, P)
    original = winding._CHUNK_ELEMS
    try:
        winding._CHUNK_ELEMS = 1000
        assert np.allclose(winding_number(mesh, P), expected)
    finally:
        winding._CHUNK_ELEMS = original


def test_output_shape():
    mesh = tetrahedron()
    assert winding_number(mesh, np.zeros((1, 3))).shape == (1,)
    assert winding_number(mesh, np.zeros((17, 3))).shape == (17,)
    with pytest.raises(ValueError):
        winding_number(mesh, np.zeros((4, 2)))


# --------------------------------------------------------- hierarchical evaluation


def _inputs_of_all_kinds():
    from gsd.shapes import humanoid, merge_meshes, mobius_strip, random_holes

    sphere = icosphere(3)
    return {
        "closed": sphere,
        "with holes": random_holes(icosphere(3), 0.2, 4, seed=2),
        "intersecting pieces": humanoid(subdivisions=1),
        "non-orientable": mobius_strip(),
        "duplicated": merge_meshes([sphere, sphere]),
        "with flipped faces": Mesh(sphere.V, np.where(
            (np.arange(sphere.n_faces) % 7 == 0)[:, None], sphere.F[:, ::-1], sphere.F)),
    }


@pytest.mark.parametrize("kind", list(_inputs_of_all_kinds()))
def test_the_hierarchy_gives_the_same_as_the_direct_sum(kind):
    """It matches up to rounding on any input, far from and right next to the
    surface."""
    from gsd.winding import WindingHierarchy

    surface = _inputs_of_all_kinds()[kind]
    rng = np.random.default_rng(0)
    lo, hi = surface.bounding_box()
    far = lo + (hi - lo) * (rng.random((1500, 3)) * 1.6 - 0.3)
    centers = surface.V[surface.F].mean(axis=1)
    close = centers + 1e-3 * rng.standard_normal(centers.shape)
    P = np.vstack([far, close])

    hierarchy = WindingHierarchy(surface, leaf_size=20)
    assert np.abs(hierarchy(P) - winding_number(surface, P)).max() < 1e-10


def test_the_closure_of_a_patch_closes_it():
    """Patch plus closure traverses each edge equally in both directions."""
    from gsd.winding import _closure

    patch = icosphere(2).F[:150]
    cap = _closure(patch)
    all_faces = np.vstack([patch, cap])
    d = np.concatenate([all_faces[:, [0, 1]], all_faces[:, [1, 2]], all_faces[:, [2, 0]]])
    sign = np.where(d[:, 0] < d[:, 1], 1, -1)
    _, inv = np.unique(np.sort(d, axis=1), axis=0, return_inverse=True)
    assert np.all(np.bincount(inv.ravel(), weights=sign) == 0)
    assert len(cap) < len(patch)
