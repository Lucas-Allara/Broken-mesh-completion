// The samples, the optimizer, the symmetry plane and the second pass against the reference values.
#include <cmath>
#include <cstdlib>
#include <string>

#include "check.h"
#include "experiment_case.h"
#include "gsd/neldermead.h"
#include "gsd/symmetry.h"
#include "reference.h"

using namespace gsd;

int main(int argc, char** argv) {
    const std::string repo = argc > 1 ? argv[1] : "..";

    // Nelder-Mead against scipy on the Rosenbrock function.
    {
        const NelderMeadResult r = nelder_mead([](const std::vector<double>& x) {
            return (1 - x[0]) * (1 - x[0]) + 100 * (x[1] - x[0] * x[0]) * (x[1] - x[0] * x[0]);
        }, {-1.2, 1.0}, 1e-8, 1e-8, 400);
        CHECK(r.iterations == reference::rosenbrock_iterations);
        CHECK_NEAR(r.x[0], reference::rosenbrock_x[0], 1e-12);
        CHECK_NEAR(r.x[1], reference::rosenbrock_x[1], 1e-12);
    }

    const experiment::Case c = experiment::spot_case(repo);

    // The samples of the surface are those of numpy.
    {
        Rng rng(0);
        const Points P = sample_surface(c.broken.V, c.broken.F, 4000, rng);
        for (int a = 0; a < 3; a++) {
            CHECK_NEAR(P[0][a], reference::case_samples[0][a], 1e-14);
            CHECK_NEAR(P[3999][a], reference::case_samples[1][a], 1e-14);
        }
    }

    // From the starting axes of the reference, its plane.
    std::vector<Vec3> axes;
    for (const auto& a : reference::case_axes) axes.push_back({a[0], a[1], a[2]});
    const Plane plane = symmetry_plane(c.broken, 4000, 0.8, 0, &axes);
    CHECK_NEAR(plane.n.x, reference::case_plane[0], 1e-10);
    CHECK_NEAR(plane.n.y, reference::case_plane[1], 1e-10);
    CHECK_NEAR(plane.n.z, reference::case_plane[2], 1e-10);
    CHECK_NEAR(plane.d, reference::case_plane[3], 1e-10);
    CHECK_NEAR(plane.residual, reference::case_plane[4], 1e-10);

    // From its own axes, the same plane up to the tolerance of the search and the sign of the normal.
    const Plane own = symmetry_plane(c.broken);
    const double sign = dot(own.n, plane.n) > 0 ? 1.0 : -1.0;
    CHECK(norm(own.n * sign - plane.n) < 1e-3);
    CHECK(std::fabs(own.d * sign - plane.d) < 1e-3);
    CHECK(own.residual < 0.5 * median(c.broken.edge_lengths()));

    // The whole reconstruction from the starting axes of the reference: its mirrored faces, and
    // after the second pass its error with symmetry.
    {
        const Reconstruction r = reconstruct_with_symmetry(c.broken, true, 0.5, 40.0, 1, &axes);
        CHECK(r.has_plane && r.has_symmetry);
        CHECK(r.mirrored.n_faces() == reference::case_mirrored);
        Rng rng(0);
        const Points missing_points = sample_surface(c.truth.V, c.missing, 100000, rng);
        const double e_method = experiment::hole_error(missing_points, r.method_V, r.method_F, rng);
        const double e_faired = experiment::hole_error(missing_points, r.faired_V, r.faired_F, rng);
        const double e_symmetry = experiment::hole_error(missing_points, r.symmetry_V, r.symmetry_F, rng);
        CHECK_NEAR(e_method, reference::case_errors[0], 1e-12);
        CHECK_NEAR(e_faired, reference::case_errors[1], 1e-9);
        CHECK_NEAR(e_symmetry, reference::case_errors[2], 1e-9);
    }

    // From its own axes the plane differs in its last digits, so the second pass is another one:
    // about the same mirrored faces, and an error below that of fairing and near the reference one.
    {
        const Reconstruction r = reconstruct_with_symmetry(c.broken);
        CHECK(r.has_plane && r.has_symmetry);
        CHECK(std::abs(r.mirrored.n_faces() - reference::case_mirrored) <= reference::case_mirrored / 20);
        Rng rng(0);
        const Points missing_points = sample_surface(c.truth.V, c.missing, 100000, rng);
        experiment::hole_error(missing_points, r.method_V, r.method_F, rng);
        const double e_faired = experiment::hole_error(missing_points, r.faired_V, r.faired_F, rng);
        const double e_symmetry = experiment::hole_error(missing_points, r.symmetry_V, r.symmetry_F, rng);
        CHECK(e_symmetry < e_faired);
        CHECK(std::fabs(e_symmetry - reference::case_errors[2]) < 0.05 * reference::case_errors[2]);
    }

    // Lucy is not symmetric: its residual exceeds half an edge, so the second pass is not run.
    {
        const Mesh lucy = load_obj(repo + "/assets/lucy.obj");
        CHECK(symmetry_plane(lucy).residual > 0.5 * median(lucy.edge_lengths()));
    }
    return check::result("symmetry");
}
