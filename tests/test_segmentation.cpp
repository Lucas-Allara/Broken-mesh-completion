// The tetrahedralization, the graph cut and the heuristics against the reference values.
#include <string>

#include "check.h"
#include "experiment_case.h"
#include "gsd/conforming.h"
#include "gsd/constraints.h"
#include "gsd/maxflow.h"
#include "gsd/segmentation.h"
#include "reference.h"

using namespace gsd;

int main(int argc, char** argv) {
    const std::string repo = argc > 1 ? argv[1] : "..";

    // numpy.percentile with linear interpolation.
    {
        std::vector<double> a(37);
        for (int i = 0; i < 37; i++) a[i] = std::sin((i + 1) * 1.7) * std::sin((i + 1) * 1.7);
        CHECK_NEAR(percentile_linear(a, 85.0), reference::percentile_85, 1e-14);
        CHECK_NEAR(median(a), reference::median_37, 1e-14);
        a.pop_back();
        CHECK_NEAR(median(a), reference::median_36, 1e-14);
    }

    // The minimum cut against all the labelings of a small problem with integer costs.
    {
        Rng rng(4);
        const int m = 12;
        std::vector<double> u_in(m), u_out(m), v;
        Adjacency adj;
        for (int i = 0; i < m; i++) { u_in[i] = static_cast<double>(rng.integers(20)); u_out[i] = static_cast<double>(rng.integers(20)); }
        for (int i = 0; i < m; i++)
            for (int j = i + 1; j < m; j++)
                if (rng.integers(3) == 0) { adj.tet_a.push_back(i); adj.tet_b.push_back(j); adj.area.push_back(1.0); v.push_back(static_cast<double>(1 + rng.integers(9))); }
        const std::vector<char> cut = min_cut_labels(u_in, u_out, adj, v);
        double best = 1e300;
        std::vector<char> x(m);
        for (int mask = 0; mask < (1 << m); mask++) {
            for (int i = 0; i < m; i++) x[i] = (mask >> i) & 1;
            best = std::min(best, energy(x, u_in, u_out, adj, v));
        }
        CHECK(energy(cut, u_in, u_out, adj, v) == best);
    }

    const experiment::Case c = experiment::spot_case(repo);
    CHECK(c.broken.n_faces() == reference::case_faces);
    const TetMesh mesh = conforming_tet_mesh(c.broken);
    const Adjacency adj = tet_adjacency(mesh);
    CHECK(mesh.n_tets() == reference::case_tets);
    CHECK(static_cast<int>(adj.tet_a.size()) == reference::case_pairs);
    // Every input face is a face of the tetrahedralization.
    const std::vector<int> inputs = input_faces(c.broken, mesh);
    CHECK(static_cast<int>(inputs.size()) == c.broken.n_faces());

    const std::vector<double> w = element_winding_number(c.broken, mesh, 1);
    CHECK_NEAR(pairwise_sum_of(w), reference::case_w_sum, 1e-11);
    std::vector<double> u_in, u_out;
    data_term(w, u_in, u_out);
    const double sigma = estimate_sigma(w, adj);
    CHECK_NEAR(sigma, reference::case_sigma, 1e-12);
    std::vector<double> v = smoothness_weights(w, adj, sigma);
    for (double& x : v) x *= 40.0;
    const std::vector<char> cut = min_cut_labels(u_in, u_out, adj, v);
    int inside_cut = 0, inside_heuristics = 0;
    for (char x : cut) inside_cut += x;
    CHECK(inside_cut == reference::case_inside_cut);
    const std::vector<char> labels = enforce_optional_constraints(mesh, inputs, w, cut, u_in, u_out, adj, v);
    for (char x : labels) inside_heuristics += x;
    CHECK(inside_heuristics == reference::case_inside_heuristics);

    Points V;
    std::vector<Tri> F;
    extract_label_boundary(mesh, labels, V, F);
    CHECK(static_cast<int>(V.size()) == reference::case_method_vertices && static_cast<int>(F.size()) == reference::case_method_faces);
    const Mesh output(V, F);
    CHECK(output.is_closed());

    // With gamma = 0 the cut is the threshold w > 0.5.
    {
        const std::vector<double> zero(v.size(), 0.0);
        const std::vector<char> threshold = min_cut_labels(u_in, u_out, adj, zero);
        int different = 0;
        for (int t = 0; t < mesh.n_tets(); t++) different += (threshold[t] != 0) != (w[t] > 0.5);
        CHECK(different == 0);
    }

    // The whole method through graphcut_segment gives the same mesh.
    {
        SegmentOptions options;
        options.gamma = 40.0;
        options.quadrature = 1;
        options.heuristics = true;
        const Segmentation s = graphcut_segment(c.broken, options);
        CHECK(s.F == F);
        Rng rng(0);
        const Points missing_points = sample_surface(c.truth.V, c.missing, 100000, rng);
        CHECK_NEAR(experiment::hole_error(missing_points, s.V, s.F, rng), reference::case_errors[0], 1e-12);
    }

    // The hard constraints, the 4-point rule and the default options.
    {
        SegmentOptions options;
        options.gamma = 40.0;
        options.quadrature = 1;
        options.hard_constraints = true;
        const Segmentation hard = graphcut_segment(c.broken, options, &mesh);
        int inside = 0;
        for (char x : hard.inside) inside += x;
        CHECK(inside == reference::case_hard_inside && static_cast<int>(hard.F.size()) == reference::case_hard_faces);
        CHECK_NEAR(pairwise_sum_of(element_winding_number(c.broken, mesh, 4)), reference::case_w4_sum, 1e-11);
        const Segmentation plain = graphcut_segment(c.broken, SegmentOptions{}, &mesh);
        inside = 0;
        for (char x : plain.inside) inside += x;
        CHECK(inside == reference::case_default_inside);
        CHECK_NEAR(plain.sigma, reference::case_default_sigma, 1e-12);
    }

    // Intersecting pieces: the faces that cross are found, and the tetrahedralization still works.
    {
        const Mesh ball = icosphere(2, 0.6);
        Points moved = ball.V;
        for (Vec3& p : moved) p = p + Vec3{0.55, 0.07, 0.03};
        const Mesh pieces = merge_meshes({ball, Mesh(moved, ball.F)});
        const std::vector<char> crossing = self_intersecting_faces(pieces);
        int n_crossing = 0;
        for (char x : crossing) n_crossing += x;
        CHECK(n_crossing == reference::pieces_crossing);
        SegmentOptions options;
        options.gamma = 40.0;
        options.quadrature = 1;
        options.heuristics = true;
        const Segmentation s = graphcut_segment(pieces, options);
        int inside = 0;
        for (char x : s.inside) inside += x;
        CHECK(s.mesh.n_tets() == reference::pieces_tets && inside == reference::pieces_inside);
        CHECK(static_cast<int>(s.F.size()) == reference::pieces_faces);
        CHECK(Mesh(s.V, s.F).is_closed());
    }
    return check::result("segmentation");
}
