// The random generator against numpy, and the meshes and the winding numbers against the
// reference values.
#include <algorithm>
#include <string>
#include <utility>

#include "check.h"
#include "gsd/mesh.h"
#include "gsd/rng.h"
#include "gsd/shapes.h"
#include "gsd/symmetry.h"
#include "gsd/winding.h"
#include "reference.h"

using namespace gsd;

int main(int argc, char** argv) {
    const std::string repo = argc > 1 ? argv[1] : "..";

    // The generator of numpy, bit for bit.
    {
        Rng r0(0), r3(3);
        for (int i = 0; i < 3; i++) CHECK(r0.random() == reference::random_0[i]);
        for (int i = 0; i < 3; i++) CHECK(r3.random() == reference::random_3[i]);
        const unsigned long long highs[4] = {5856ull, 12000ull, 7ull, 1000000000000ull};
        for (int i = 0; i < 4; i++) CHECK(static_cast<long long>(r0.integers(highs[i])) == reference::integers_0[i]);
        for (int i = 0; i < 4; i++) CHECK(static_cast<long long>(r3.integers(highs[i])) == reference::integers_3[i]);
        std::vector<double> p = {0.1, 0.5, 0.25, 0.15, 3.0, 1.0};
        double total = 0.0;
        for (double v : p) total += v;
        for (double& v : p) v /= total;
        const std::vector<int> c0 = r0.choice(p, 8), c3 = r3.choice(p, 8);
        for (int i = 0; i < 8; i++) CHECK(c0[i] == reference::choice_0[i] && c3[i] == reference::choice_3[i]);
    }

    // A closed sphere: 1 inside, 0 outside, -1 inside when its orientation is flipped.
    {
        const Mesh sphere = icosphere(3);
        CHECK(sphere.is_closed() && sphere.n_faces() == 1280 && connected_components(sphere) == 1);
        const std::vector<double> w = winding_number(sphere, {{0, 0, 0}, {0.3, -0.2, 0.5}, {2, 0, 0}});
        CHECK_NEAR(w[0], 1.0, 1e-12);
        CHECK_NEAR(w[1], 1.0, 1e-12);
        CHECK_NEAR(w[2], 0.0, 1e-12);
        std::vector<Tri> flipped = sphere.F;
        for (Tri& t : flipped) std::swap(t[1], t[2]);
        CHECK_NEAR(winding_number(Mesh(sphere.V, flipped), {{0, 0, 0}})[0], -1.0, 1e-12);
        // With a hole, w at the center is close to one minus the fraction of the sphere that is missing.
        const Mesh open = punch_hole(sphere, {0, 0, 1}, 0.6);
        const double missing = 1.0 - open.total_area() / sphere.total_area();
        CHECK_NEAR(winding_number(open, {{0, 0, 0}})[0], 1.0 - missing, 5e-3);
        CHECK(!open.is_closed() && !open.boundary_edges().empty());
    }

    // Spot with random holes: the holes of the reference values, and its winding numbers.
    {
        const Mesh spot = load_obj(repo + "/assets/spot/spot_triangulated.obj");
        CHECK(spot.n_faces() == 5856 && spot.is_closed());
        CHECK_NEAR(spot.total_area(), reference::spot_area, 1e-14);
        const int seeds[3] = {3, 7, 21};
        Mesh broken;
        for (int k = 0; k < 3; k++) {
            const Mesh h = random_holes(spot, 0.10, 5, seeds[k]);
            long long sum = 0;
            for (const Tri& t : h.F) sum += t[0] + 3LL * t[1] + 7LL * t[2];
            CHECK(h.n_faces() == reference::holes_faces[k]);
            CHECK(sum == reference::holes_checksum[k]);
            if (k == 0) broken = h;
        }
        Points P;
        for (const auto& p : reference::winding_points) P.push_back({p[0], p[1], p[2]});
        const std::vector<double> direct = winding_number(broken, P);
        const WindingHierarchy hierarchy(broken);
        const std::vector<double> fast = hierarchy(P);
        for (std::size_t i = 0; i < P.size(); i++) {
            CHECK_NEAR(direct[i], reference::winding_direct[i], 1e-13);
            CHECK_NEAR(fast[i], reference::winding_hierarchy[i], 1e-13);
        }
        // The hierarchy gives the direct sum at many points: 2000 in the box of the model and 1000
        // next to the surface, at distances from 1e-6 to 1e-2 on both sides.
        Rng rng(11);
        Vec3 lo, hi;
        spot.bounds(lo, hi);
        Points Q(2000);
        for (Vec3& q : Q)
            for (int a = 0; a < 3; a++) q[a] = lo[a] - 0.05 + (hi[a] - lo[a] + 0.1) * rng.random();
        for (const Vec3& p : sample_surface(broken.V, broken.F, 1000, rng)) {
            const double distance = std::pow(10.0, -6.0 + 4.0 * rng.random()) * (rng.random() < 0.5 ? -1.0 : 1.0);
            Vec3 direction{rng.random() - 0.5, rng.random() - 0.5, rng.random() - 0.5};
            Q.push_back(p + direction * (distance / norm(direction)));
        }
        const std::vector<double> wd = winding_number(broken, Q), wh = hierarchy(Q);
        double worst = 0.0;
        for (std::size_t i = 0; i < Q.size(); i++) worst = std::max(worst, std::fabs(wd[i] - wh[i]));
        CHECK(worst < 1e-12);
    }

    // Splitting edges keeps the surface: same area, no hanging vertices on a closed mesh.
    {
        const Mesh sphere = icosphere(2);
        std::vector<char> split(sphere.n_edges());
        for (int e = 0; e < sphere.n_edges(); e++) split[e] = e % 3 == 0 || e % 7 == 1;
        const Mesh fine = sphere.split_edges(split);
        CHECK(fine.is_closed());
        CHECK_NEAR(fine.total_area(), sphere.total_area(), 1e-12);
        CHECK(fine.n_vertices() > sphere.n_vertices());
    }
    return check::result("winding");
}
