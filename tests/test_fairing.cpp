// The fairing of the caps against the reference values and against the reference error.
#include <string>

#include "check.h"
#include "experiment_case.h"
#include "gsd/fairing.h"
#include "gsd/segmentation.h"
#include "reference.h"

using namespace gsd;

int main(int argc, char** argv) {
    const std::string repo = argc > 1 ? argv[1] : "..";
    const experiment::Case c = experiment::spot_case(repo);
    SegmentOptions options;
    options.gamma = 40.0;
    options.quadrature = 1;
    options.heuristics = true;
    const Segmentation s = graphcut_segment(c.broken, options);

    Points V;
    std::vector<Tri> F;
    fair_filled_patches(s.V, s.F, c.broken, V, F);
    CHECK(static_cast<int>(V.size()) == reference::case_faired_vertices && static_cast<int>(F.size()) == reference::case_faired_faces);
    Vec3 sum{0, 0, 0};
    for (const Vec3& p : V) sum += p;
    for (int a = 0; a < 3; a++) CHECK_NEAR(sum[a], reference::case_faired_sum[a], 1e-9);
    CHECK(Mesh(V, F).is_closed());

    // The input is not touched: its vertices are still there, bit for bit.
    Vec3 lo, hi;
    c.broken.bounds(lo, hi);
    const std::vector<char> on = on_input(V, c.broken, 1e-9 * norm(hi - lo));
    int n_on = 0;
    for (char x : on) n_on += x;
    CHECK(n_on >= c.broken.n_vertices());
    for (int i = 0; i < 50; i++) {
        const Vec3& p = c.broken.V[(i * 37) % c.broken.n_vertices()];
        bool found = false;
        for (const Vec3& q : V) found |= q.x == p.x && q.y == p.y && q.z == p.z;
        CHECK(found);
    }

    // The errors of the experiment for this case, as in the reference values.
    Rng rng(0);
    const Points missing_points = sample_surface(c.truth.V, c.missing, 100000, rng);
    const double e_method = experiment::hole_error(missing_points, s.V, s.F, rng);
    const double e_faired = experiment::hole_error(missing_points, V, F, rng);
    CHECK_NEAR(e_method, reference::case_errors[0], 1e-12);
    CHECK_NEAR(e_faired, reference::case_errors[1], 1e-9);
    CHECK(e_faired < e_method);

    // A mesh without caps comes back as it is.
    {
        const Mesh sphere = icosphere(2);
        Points V2;
        std::vector<Tri> F2;
        fair_filled_patches(sphere.V, sphere.F, sphere, V2, F2);
        CHECK(F2 == sphere.F);
    }
    // The Laplacian (order 1) is a different surface from the tri-Laplacian, and by the maximum
    // principle its caps stay inside the box of the input.
    {
        Points V1;
        std::vector<Tri> F1;
        fair_filled_patches(s.V, s.F, c.broken, V1, F1, 2, 3, 1);
        CHECK(Mesh(V1, F1).is_closed());
        bool inside = true;
        Vec3 sum1{0, 0, 0};
        for (const Vec3& p : V1) {
            sum1 += p;
            for (int a = 0; a < 3; a++) inside &= p[a] >= lo[a] - 1e-12 && p[a] <= hi[a] + 1e-12;
        }
        CHECK(inside);
        CHECK(norm(sum1 - sum) > 1e-3);
    }
    return check::result("fairing");
}
