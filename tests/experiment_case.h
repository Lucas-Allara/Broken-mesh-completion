// The case of the experiment that the tests use: Spot with 10% of its surface missing, seed 3,
// built as apps/gsd_experiment.cpp builds it.
#pragma once
#include <algorithm>
#include <string>
#include <unordered_set>

#include "gsd/kdtree.h"
#include "gsd/mesh.h"
#include "gsd/rng.h"
#include "gsd/shapes.h"
#include "gsd/symmetry.h"

namespace experiment {

const double kSize = 1.35;

struct Case {
    gsd::Mesh truth, broken;
    std::vector<gsd::Tri> missing;
};

inline Case spot_case(const std::string& repo, double fraction = 0.10, int seed = 3) {
    const gsd::Mesh m = gsd::load_obj(repo + "/assets/spot/spot_triangulated.obj");
    gsd::Vec3 lo, hi;
    m.bounds(lo, hi);
    const gsd::Vec3 center = (lo + hi) / 2.0;
    const double scale = kSize / std::max(hi.x - lo.x, std::max(hi.y - lo.y, hi.z - lo.z));
    gsd::Points V(m.V.size());
    for (std::size_t i = 0; i < V.size(); i++) {
        const gsd::Vec3 p = (m.V[i] - center) * scale;
        V[i] = {p.x, -p.z, p.y};
    }
    Case c;
    c.truth = gsd::Mesh(std::move(V), m.F);
    const gsd::Mesh holed = gsd::random_holes(c.truth, fraction, 5, seed);
    c.broken = gsd::compact(holed.V, holed.F);
    const std::unordered_set<gsd::Tri, gsd::TriHash> present(holed.F.begin(), holed.F.end());
    for (const gsd::Tri& t : c.truth.F)
        if (!present.count(t)) c.missing.push_back(t);
    return c;
}

inline double hole_error(const gsd::Points& missing_points, const gsd::Points& V, const std::vector<gsd::Tri>& F, gsd::Rng& rng) {
    const gsd::KdTree tree(gsd::sample_surface(V, F, 300000, rng));
    const std::vector<double> d = tree.nearest_distances(missing_points);
    return 100.0 * (gsd::pairwise_sum_of(d) / static_cast<double>(d.size())) / kSize;
}

}  // namespace experiment
