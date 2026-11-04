"""Segmentation by graph cut (Section 5) and tetrahedralizations."""

from itertools import product

import numpy as np
import pytest

from gsd.conforming import (
    conforming_tet_mesh,
    delaunay_tet_mesh,
    grid_tet_mesh,
    self_intersecting_faces,
)
from gsd.mesh import Mesh
from gsd.segmentation import (
    apply_hard_constraints,
    data_term,
    element_winding_number,
    estimate_sigma,
    extract_label_boundary,
    graphcut_segment,
    hard_constraints_from_surface,
    min_cut_labels,
    smoothness_weights,
    tet_adjacency,
)
from gsd.shapes import humanoid, icosphere, merge_meshes, random_holes, torus
from gsd.winding import winding_number
from gsd.tetmesh import tet_grid


# --------------------------------------------------------- tetrahedron adjacency


def test_neighbors_really_share_a_face():
    """Each pair shares three vertices and no interior face is missing."""
    mesh = tet_grid(n=4, size=2.0)
    tet_a, tet_b, area = tet_adjacency(mesh)

    assert len(tet_a) == int((mesh.face_tet_count == 2).sum())
    assert (area > 0).all()

    vertex_sets = [set(map(int, mesh.T[t])) for t in range(mesh.n_tets)]
    for a, b in zip(tet_a, tet_b):
        assert len(vertex_sets[a] & vertex_sets[b]) == 3


def test_the_neighborhood_area_is_that_of_the_shared_face():
    """``a_ij`` is the area of the shared face."""
    mesh = tet_grid(n=3, size=2.0)
    tet_a, tet_b, area = tet_adjacency(mesh)

    for k in range(0, len(tet_a), 37):
        shared = set(map(int, mesh.T[tet_a[k]])) & set(map(int, mesh.T[tet_b[k]]))
        v = mesh.V[sorted(shared)]
        direct_area = 0.5 * np.linalg.norm(
            np.cross(v[1] - v[0], v[2] - v[0])
        )
        assert abs(area[k] - direct_area) < 1e-9


# --------------------------------------------------------------- the energy


def test_data_term_equation_9():
    w = np.array([-0.3, 0.0, 0.3, 0.5, 0.7, 1.0, 1.4])
    u_inside, u_outside = data_term(w)

    assert np.allclose(u_inside, np.maximum(1 - w, 0))
    assert np.allclose(u_outside, np.maximum(w, 0))
    assert u_inside[w == 1.0] == 0.0
    assert u_outside[w == 0.0] == 0.0


def test_smoothness_term_decays_with_the_difference():
    """Separating two neighbors costs less the more different their w is."""
    tet_a = np.array([0, 0, 0])
    tet_b = np.array([1, 1, 1])
    area = np.array([1.0, 1.0, 1.0])

    w = np.array([0.5, 0.5])
    w2 = np.array([0.55, 0.9])
    v1 = smoothness_weights(w, tet_a[:1], tet_b[:1], area[:1], sigma=0.2)
    v2 = smoothness_weights(w2, tet_a[:1], tet_b[:1], area[:1], sigma=0.2)
    assert v1[0] > v2[0]
    assert (smoothness_weights(np.array([0.5, 0.5]), np.array([0]), np.array([1]),
                               np.array([3.7]), sigma=0.2) <= 3.7 + 1e-12).all()


def test_estimate_sigma_is_positive_and_does_not_collapse_with_constant_w():
    tet_a, tet_b = np.array([0, 1]), np.array([1, 2])
    assert estimate_sigma(np.array([0.4, 0.4, 0.4]), tet_a, tet_b) > 0

    w = np.array([0.1, 0.5, 0.9])
    sigma = estimate_sigma(w, tet_a, tet_b, percentile=85.0)
    assert sigma == pytest.approx(
        np.percentile(np.abs(w[tet_a] - w[tet_b]), 85.0)
    )


def test_estimate_sigma_is_not_dominated_by_the_far_majority():
    """With most pairs equal, the median comes out tiny and the
    percentile does not."""
    rng = np.random.default_rng(5)
    far = rng.normal(0, 0.002, 700)
    near = rng.uniform(0.1, 0.6, 300)
    w = np.concatenate([np.zeros(1), far, near])
    tet_a = np.zeros(1000, dtype=np.int64)
    tet_b = np.arange(1, 1001, dtype=np.int64)

    median_diff = np.median(np.abs(w[tet_a] - w[tet_b]))
    sigma = estimate_sigma(w, tet_a, tet_b)

    assert median_diff < 0.01
    assert sigma > 0.05
    assert sigma > 50 * median_diff


# ---------------------------------------------------- the minimum cut itself


def _energy(x, u_inside, u_outside, tet_a, tet_b, v):
    """Equation 8 evaluated at the labeling ``x``."""
    x = np.asarray(x, dtype=bool)
    cost = np.where(x, u_inside, u_outside).sum()
    cost += v[x[tet_a] != x[tet_b]].sum()
    return float(cost)


def test_the_minimum_cut_matches_brute_force():
    """Against the enumeration of the 2^m labelings."""
    rng = np.random.default_rng(0)
    for _ in range(30):
        m = int(rng.integers(3, 9))
        u_inside = rng.uniform(0, 2, m)
        u_outside = rng.uniform(0, 2, m)

        n_pairs = int(rng.integers(1, 2 * m))
        a = rng.integers(0, m, n_pairs)
        b = rng.integers(0, m, n_pairs)
        distinct = a != b
        a, b = a[distinct], b[distinct]
        v = rng.uniform(0, 3, len(a))

        labels = min_cut_labels(u_inside, u_outside, a, b, v)
        found = _energy(labels, u_inside, u_outside, a, b, v)
        optimal = min(
            _energy(x, u_inside, u_outside, a, b, v)
            for x in product([False, True], repeat=m)
        )

        assert found == pytest.approx(optimal, abs=1e-3)


def test_gamma_zero_is_exactly_the_threshold():
    """Without the smoothness term it gives the threshold ``w > 0.5`` (Equation 7)."""
    rng = np.random.default_rng(1)
    w = rng.uniform(-0.4, 1.4, size=60)
    u_inside, u_outside = data_term(w)

    a = rng.integers(0, 60, 150)
    b = rng.integers(0, 60, 150)
    v_zero = np.zeros(150)

    labels = min_cut_labels(u_inside, u_outside, a, b, v_zero)
    assert np.array_equal(labels, w > 0.5)


def test_hard_constraint_forces_the_label_whatever_the_data():
    """The constraints are respected and, among the labelings that satisfy them,
    the cut gives the one with the lowest energy."""
    rng = np.random.default_rng(2)
    tried = 0
    while tried < 20:
        m = int(rng.integers(3, 8))
        u_inside = rng.uniform(0, 2, m)
        u_outside = rng.uniform(0, 2, m)
        n_pairs = int(rng.integers(1, 2 * m))
        a = rng.integers(0, m, n_pairs)
        b = rng.integers(0, m, n_pairs)
        distinct = a != b
        a, b = a[distinct], b[distinct]
        v = rng.uniform(0, 3, len(a))

        assignment = rng.integers(-1, 2, m)  # -1 free, 0 force outside, 1 force inside
        force_inside = assignment == 1
        force_outside = assignment == 0
        if not force_inside.any() and not force_outside.any():
            continue
        tried += 1

        ui2, uo2 = apply_hard_constraints(u_inside, u_outside, v,
                                          force_inside, force_outside)
        labels = min_cut_labels(ui2, uo2, a, b, v)

        assert np.all(labels[force_inside])
        assert not np.any(labels[force_outside])

        compatible = [
            x for x in product([False, True], repeat=m)
            if all(np.asarray(x)[force_inside]) and not any(np.asarray(x)[force_outside])
        ]
        optimal = min(_energy(np.asarray(x), u_inside, u_outside, a, b, v)
                      for x in compatible)
        found = _energy(labels, u_inside, u_outside, a, b, v)
        assert found == pytest.approx(optimal, abs=1e-3)


def test_contradictory_constraints_raise_an_error():
    u_inside = np.array([1.0, 1.0])
    u_outside = np.array([1.0, 1.0])
    v = np.array([0.5])
    with pytest.raises(ValueError):
        apply_hard_constraints(u_inside, u_outside, v,
                               force_inside=np.array([True, False]),
                               force_outside=np.array([True, False]))


def test_smoothness_corrects_incoherent_noise_between_neighbors():
    """With noisy w around a transition, the cut improves on the threshold as
    γ grows."""
    mesh = tet_grid(n=8, size=2.0)
    tet_a, tet_b, area = tet_adjacency(mesh)
    centers = mesh.tet_centroids

    w_true = 0.5 * (1 - np.tanh(centers[:, 0] / 0.35))
    rng = np.random.default_rng(3)
    w_noisy = np.clip(w_true + rng.normal(0, 0.22, size=len(w_true)), -0.4, 1.4)
    truth = centers[:, 0] < 0

    u_inside, u_outside = data_term(w_noisy)
    sigma = estimate_sigma(w_noisy, tet_a, tet_b)

    acc_threshold = np.mean((w_noisy > 0.5) == truth)
    assert acc_threshold < 0.96

    gammas = (0.0, 1.0, 5.0, 15.0, 40.0)
    accuracies = []
    for gamma in gammas:
        v = gamma * smoothness_weights(w_noisy, tet_a, tet_b, area, sigma)
        labels = min_cut_labels(u_inside, u_outside, tet_a, tet_b, v)
        accuracies.append(np.mean(labels == truth))

    assert accuracies[0] == pytest.approx(acc_threshold, abs=1e-9)
    assert all(p2 >= p1 - 1e-9 for p1, p2 in zip(accuracies, accuracies[1:]))
    assert accuracies[-1] - accuracies[0] > 0.03


# ------------------------------------------------------- surface extraction


def test_the_extracted_boundary_is_manifold_and_closed():
    """On a ball of tetrahedra the boundary is closed and faces outward."""
    mesh = tet_grid(n=6, size=2.0)
    centers = mesh.tet_centroids
    inside = np.linalg.norm(centers, axis=1) < 0.7

    V, F = extract_label_boundary(mesh, inside)
    used_ids = np.unique(F)
    sub = Mesh(V[used_ids], np.searchsorted(used_ids, F))
    assert sub.is_closed
    assert len(F) > 0

    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    volume = np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0
    assert volume > 0


# ------------------------------------------------- interchangeable discretizations


def _interlocking_spheres() -> Mesh:
    """Two interpenetrating spheres, displaced in a generic direction."""
    a = icosphere(2, radius=0.6)
    b = icosphere(2, radius=0.6)
    return merge_meshes([a, Mesh(b.V + np.array([0.55, 0.07, 0.03]), b.F)])


def _brute_force_intersections(surf: Mesh) -> np.ndarray:
    """Self-intersections by trying every pair."""
    from gsd.conforming import _edge_crosses

    T = surf.V[surf.F]
    i, j = np.triu_indices(len(T), k=1)
    crossing = np.zeros(len(i), dtype=bool)
    for k in range(3):
        nxt = (k + 1) % 3
        crossing |= _edge_crosses(T[i, k], T[i, nxt], T[j, 0], T[j, 1], T[j, 2])
        crossing |= _edge_crosses(T[j, k], T[j, nxt], T[i, 0], T[i, 1], T[i, 2])
    mask = np.zeros(len(T), dtype=bool)
    mask[i[crossing]] = True
    mask[j[crossing]] = True
    return mask


def test_an_intact_mesh_has_no_self_intersections():
    """Neighboring faces touch but do not intersect."""
    for intact in (icosphere(3), torus(24, 12)):
        assert not self_intersecting_faces(intact).any()


def test_self_intersections_hand_built_cases():
    """Two crossed triangles intersect and a distant one does not. A large triangle
    crossed by a small one does too (they fall in different radius classes)."""
    V = np.array([
        [-1, -1, 0], [1, -1, 0], [0, 1, 0],          # horizontal
        [0, -0.2, -1], [0, 0.2, -1], [0, 0, 1],      # vertical, crosses it
        [5, 5, 5], [6, 5, 5], [5, 6, 5],             # far away
        [-30, -30, 2], [30, -30, 2], [0, 30, 2],     # huge, at z = 2
        [0, 0, 1.9], [0.1, 0, 2.1], [0, 0.1, 2.1],   # small, crosses z = 2
    ], dtype=float)
    F = np.arange(15).reshape(5, 3)
    mask = self_intersecting_faces(Mesh(V, F))
    assert mask.tolist() == [True, True, False, True, True]


def test_the_pair_search_misses_no_intersection():
    surf = _interlocking_spheres()
    expected = _brute_force_intersections(surf)
    obtained = self_intersecting_faces(surf)
    assert expected.sum() > 0
    assert np.array_equal(obtained, expected)


def test_red_green_refinement_is_conforming_and_does_not_move_the_surface():
    """The sphere stays closed, with the same area and volume."""
    from gsd.conforming import _red_green_refine

    sphere = icosphere(2)
    rng = np.random.default_rng(3)
    marked = rng.random(sphere.n_faces) < 0.1
    V, F = _red_green_refine(sphere.V, sphere.F, marked)
    fine = Mesh(V, F)

    assert fine.is_closed
    assert fine.total_area == pytest.approx(sphere.total_area, rel=1e-12)

    def volume(m):
        a, b, c = m.V[m.F[:, 0]], m.V[m.F[:, 1]], m.V[m.F[:, 2]]
        return np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0

    assert volume(fine) == pytest.approx(volume(sphere), rel=1e-12)
    assert fine.n_faces > sphere.n_faces + 3 * marked.sum()


def test_automatic_constraints_invent_nothing_on_a_grid():
    """On a grid no input face is a face of the mesh."""
    broken = random_holes(icosphere(2), target_fraction=0.15, n_holes=4, seed=7)
    force_inside, force_outside = hard_constraints_from_surface(
        broken, grid_tet_mesh(broken, n=9))
    assert not force_inside.any()
    assert not force_outside.any()


def _segment_broken_sphere(make_mesh):
    """Segments the sphere with holes on ``make_mesh(broken)`` and checks
    that the mesh covers the input and the output is closed."""
    broken = random_holes(icosphere(2), target_fraction=0.15, n_holes=4, seed=7)
    mesh = make_mesh(broken)

    lo, hi = broken.bounding_box()
    assert np.all(mesh.V.min(axis=0) <= lo) and np.all(mesh.V.max(axis=0) >= hi)

    result = graphcut_segment(broken, mesh=mesh, gamma=3.0)
    assert Mesh(result["V"], result["F"]).is_closed
    return mesh, result


def test_segmentation_runs_on_unconstrained_delaunay():
    """Delaunay adds no points: the output lies exactly on the sphere."""
    _, result = _segment_broken_sphere(delaunay_tet_mesh)
    deviation = np.abs(np.linalg.norm(result["V"], axis=1) - 1.0)
    assert deviation.max() < 1e-12


def test_segmentation_runs_on_a_grid():
    """On a grid the output departs from the sphere by the order of one
    cell (more over the holes)."""
    n = 14
    mesh, result = _segment_broken_sphere(lambda s: grid_tet_mesh(s, n=n))
    cell = float((mesh.V.max(axis=0) - mesh.V.min(axis=0)).max()) / n
    deviation = np.abs(np.linalg.norm(result["V"], axis=1) - 1.0)
    assert deviation.mean() < 0.6 * cell


# ---------------------------------------------------------------- full pipeline


tetgen = pytest.importorskip("tetgen", reason="the full pipeline needs TetGen")


def test_conforming_tet_mesh_preserves_every_input_triangle():
    """Each input triangle is a face of the mesh, with the same
    vertices."""
    sphere = icosphere(3, radius=0.5)
    mesh, ids = conforming_tet_mesh(sphere)

    d = np.linalg.norm(mesh.V[ids] - sphere.V, axis=1)
    assert d.max() < 1e-12

    mesh_faces = {tuple(c) for c in np.sort(mesh.faces, axis=1)}
    missing = sum(
        tuple(sorted(int(ids[v]) for v in tri)) not in mesh_faces
        for tri in np.asarray(sphere.F)
    )
    assert missing == 0


def test_graphcut_segment_reconstructs_a_sphere_with_holes():
    intact = icosphere(2)
    broken = random_holes(intact, target_fraction=0.15, n_holes=4, seed=7)

    result = graphcut_segment(broken, gamma=3.0, margin=0.22, max_far_edge=0.35)
    assert len(result["F"]) > 0

    area = Mesh(result["V"], result["F"]).total_area
    assert 0.85 < area / intact.total_area < 1.30


def test_hard_constraints_from_surface_do_not_overlap_and_mark_the_intact_part():
    """The constraints do not overlap, and they agree with what w already says."""
    intact = icosphere(2)
    broken = random_holes(intact, target_fraction=0.15, n_holes=4, seed=7)
    mesh, _ = conforming_tet_mesh(broken, margin=0.22, max_far_edge=0.35)

    force_inside, force_outside = hard_constraints_from_surface(broken, mesh)
    assert not np.any(force_inside & force_outside)
    assert force_inside.sum() > 0
    assert force_outside.sum() > 0

    w = element_winding_number(broken, mesh)
    assert w[force_inside].mean() > 0.9
    assert w[force_outside].mean() < 0.1


def test_hard_constraints_do_not_worsen_the_reconstructed_area():
    intact = icosphere(2)
    broken = random_holes(intact, target_fraction=0.15, n_holes=4, seed=7)
    mesh, _ = conforming_tet_mesh(broken, margin=0.22, max_far_edge=0.35)

    free = graphcut_segment(broken, mesh=mesh, gamma=3.0)
    forced = graphcut_segment(broken, mesh=mesh, gamma=3.0, hard_constraints=True)

    area_free = Mesh(free["V"], free["F"]).total_area / intact.total_area
    area_forced = Mesh(forced["V"], forced["F"]).total_area / intact.total_area
    assert abs(area_forced - 1.0) < abs(area_free - 1.0) + 0.05


def test_extracting_twice_with_the_same_mesh_is_deterministic():
    intact = icosphere(1)
    mesh, _ = conforming_tet_mesh(intact, margin=0.25, max_far_edge=0.4)

    r1 = graphcut_segment(intact, mesh=mesh, gamma=2.0)
    r2 = graphcut_segment(intact, mesh=mesh, gamma=2.0)
    assert np.array_equal(r1["inside"], r2["inside"])


@pytest.mark.parametrize("rounds", [0, 2])
def test_conforming_tet_mesh_accepts_self_intersections(rounds):
    """With intersecting faces, the ones that do not intersect are still faces of the
    mesh. When subdividing, the neighbors of the intersecting ones are bisected and
    nothing is required of those."""
    surf = _interlocking_spheres()
    crossing = self_intersecting_faces(surf)
    mesh, ids = conforming_tet_mesh(surf, refine_intersections=rounds)

    assert np.abs(mesh.V[ids] - surf.V).max() == 0.0
    mesh_faces = {tuple(c) for c in np.sort(mesh.faces, axis=1)}
    present = np.array([tuple(sorted(int(ids[v]) for v in tri)) in mesh_faces
                        for tri in surf.F])
    if rounds == 0:
        assert present[~crossing].all()
    else:
        touched = np.isin(surf.F, surf.F[crossing]).any(axis=1)
        assert present[~touched].all()


def _correct_boundary_fraction(surf: Mesh, result: dict) -> float:
    """Fraction of the output area with ``w > ½`` behind and ``w < ½`` in front.
    A membrane in the air or a buried shell does not satisfy it."""
    output = Mesh(result["V"], result["F"])
    centers = output.V[output.F].mean(axis=1)
    step = 0.05 * surf.mean_edge_length * output.face_normals
    front = winding_number(surf, centers + step)
    behind = winding_number(surf, centers - step)
    good = (behind > 0.5) & (front < 0.5)
    return float(output.face_areas[good].sum() / output.total_area)


def test_subdividing_the_intersections_removes_the_membranes():
    """On the humanoid, without subdividing there are membranes between the limbs."""
    hum = humanoid(subdivisions=1)
    lo, hi = hum.bounding_box()
    far = float(np.linalg.norm(hi - lo)) / 15

    fractions = []
    for rounds in (0, 2):
        mesh, _ = conforming_tet_mesh(hum, max_far_edge=far,
                                      refine_intersections=rounds)
        result = graphcut_segment(hum, mesh=mesh, gamma=3.0, quadrature=1)
        fractions.append(_correct_boundary_fraction(hum, result))

    assert fractions[1] > 0.95
    assert fractions[1] > fractions[0] + 0.1


def test_interlocking_pieces_give_the_union():
    """The output is the boundary of the union, and the automatic constraints
    do not change it."""
    surf = _interlocking_spheres()
    mesh, _ = conforming_tet_mesh(surf, max_far_edge=0.05)

    free = graphcut_segment(surf, mesh=mesh, gamma=3.0)
    # No boundaries, although along the intersection curve there can be edges
    # with four faces.
    F = free["F"]
    edges = np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1)
    _, valence = np.unique(edges, axis=0, return_counts=True)
    assert np.all(valence % 2 == 0)
    assert _correct_boundary_fraction(surf, free) > 0.98

    r, d = 0.6, float(np.linalg.norm([0.55, 0.07, 0.03]))
    cap = 2 * np.pi * r * (r - d / 2)
    union = 2 * (4 * np.pi * r ** 2) - 2 * cap
    assert Mesh(free["V"], free["F"]).total_area == pytest.approx(union, rel=0.03)

    forced = graphcut_segment(surf, mesh=mesh, gamma=3.0, hard_constraints=True)
    assert np.array_equal(forced["inside"], free["inside"])


def test_the_heuristics_of_section_5_2():
    """On the CDT of the convex hull, the cut leaves non-manifold edges and
    input faces outside. The heuristics fix both things."""
    from gsd.constraints import input_faces

    broken = random_holes(icosphere(3), target_fraction=0.15, n_holes=4, seed=7)
    used, F = np.unique(broken.F, return_inverse=True)
    broken = Mesh(broken.V[used], F.reshape(-1, 3))
    mesh, _ = conforming_tet_mesh(broken)
    inputs = input_faces(broken, mesh)

    free = graphcut_segment(broken, mesh=mesh, gamma=3.0, quadrature=1)
    with_h = graphcut_segment(broken, mesh=mesh, gamma=3.0, quadrature=1, heuristics=True)
    n_free = np.bincount(mesh.tet_faces[free["inside"]].ravel(), minlength=mesh.n_faces)
    n_with = np.bincount(mesh.tet_faces[with_h["inside"]].ravel(), minlength=mesh.n_faces)

    assert not Mesh(free["V"], free["F"]).is_closed
    assert (n_free[inputs] == 0).any()
    assert Mesh(with_h["V"], with_h["F"]).is_closed
    assert (n_with[inputs] > 0).all()


def test_the_heuristics_do_not_touch_a_valid_output():
    intact = icosphere(2)
    mesh, _ = conforming_tet_mesh(intact, max_far_edge=0.35)
    free = graphcut_segment(intact, mesh=mesh, gamma=3.0)
    with_h = graphcut_segment(intact, mesh=mesh, gamma=3.0, heuristics=True)
    assert np.array_equal(free["inside"], with_h["inside"])
