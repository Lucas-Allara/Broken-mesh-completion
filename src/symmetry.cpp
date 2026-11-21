#include "gsd/symmetry.h"

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

#include "gsd/conforming.h"
#include "gsd/fairing.h"
#include "gsd/kdtree.h"
#include "gsd/neldermead.h"
#include "gsd/segmentation.h"

namespace gsd {

Points sample_surface(const Points& V, const std::vector<Tri>& F, int k, Rng& rng) {
    std::vector<double> area(F.size());
    for (std::size_t f = 0; f < F.size(); f++) area[f] = 0.5 * norm(cross(V[F[f][1]] - V[F[f][0]], V[F[f][2]] - V[F[f][0]]));
    const double total = F.empty() ? 0.0 : pairwise_sum_of(area);
    if (!(total > 0)) throw std::invalid_argument("a surface without area cannot be sampled");
    for (double& a : area) a /= total;
    const std::vector<int> face = rng.choice(area, k);
    std::vector<double> u(k), v(k);
    for (int i = 0; i < k; i++) u[i] = rng.random();
    for (int i = 0; i < k; i++) v[i] = rng.random();
    Points out(k);
    for (int i = 0; i < k; i++) {
        if (u[i] + v[i] > 1) { u[i] = 1 - u[i]; v[i] = 1 - v[i]; }
        const Tri& t = F[face[i]];
        out[i] = V[t[0]] + u[i] * (V[t[1]] - V[t[0]]) + v[i] * (V[t[2]] - V[t[0]]);
    }
    return out;
}

double median(std::vector<double> values) {
    const std::size_t n = values.size();
    if (n == 0) throw std::invalid_argument("median of an empty array");
    std::sort(values.begin(), values.end());
    return n % 2 ? values[n / 2] : (values[n / 2 - 1] + values[n / 2]) / 2.0;
}

namespace {

// Reflection of p across the plane n . x = d. The product p . n is rounded as the BLAS kernel of
// numpy rounds it on x86-64 (fused, y first), so that the mirrored vertices have the coordinates
// behind the reference values of the tests to the last bit: the second pass is sensitive to them.
Vec3 reflect(const Vec3& p, const Vec3& n, double d) {
    const double pn = std::fma(p.z, n.z, std::fma(p.x, n.x, p.y * n.y));
    return p - 2.0 * (pn - d) * n;
}

Vec3 normal_of(double theta, double phi) { return {std::cos(theta) * std::cos(phi), std::sin(theta) * std::cos(phi), std::sin(phi)}; }

}  // namespace

Plane symmetry_plane(const Mesh& surf, int n_samples, double trim, std::uint64_t seed, const std::vector<Vec3>* given_axes) {
    Rng rng(seed);
    const Points P = sample_surface(surf.V, surf.F, n_samples, rng);
    const int n_tree = std::min(std::max(50000, 10 * surf.n_faces()), 300000);
    const KdTree tree(sample_surface(surf.V, surf.F, n_tree, rng));
    const int cut = static_cast<int>(trim * static_cast<double>(P.size()));

    Points reflected(P.size());
    auto cost = [&](const std::vector<double>& x) {
        const Vec3 n = normal_of(x[0], x[1]);
        for (std::size_t i = 0; i < P.size(); i++) reflected[i] = reflect(P[i], n, x[2]);
        std::vector<double> d = tree.nearest_distances(reflected);
        std::nth_element(d.begin(), d.begin() + cut, d.end());
        std::sort(d.begin(), d.begin() + cut);
        return pairwise_sum(d.data(), cut) / cut;
    };

    Vec3 center{0, 0, 0};
    for (const Vec3& p : P) center += p;
    center = center / static_cast<double>(P.size());

    std::vector<Vec3> axes;
    if (given_axes) {
        axes = *given_axes;
    } else {
        // Principal axes of the samples, from the largest spread to the smallest, each with its
        // largest component positive.
        Eigen::Matrix3d C = Eigen::Matrix3d::Zero();
        for (const Vec3& p : P) {
            const Eigen::Vector3d q(p.x - center.x, p.y - center.y, p.z - center.z);
            C += q * q.transpose();
        }
        const Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(C);
        for (int k = 2; k >= 0; k--) {
            Eigen::Vector3d a = solver.eigenvectors().col(k);
            int big = 0;
            for (int i = 1; i < 3; i++)
                if (std::fabs(a[i]) > std::fabs(a[big])) big = i;
            if (a[big] < 0) a = -a;
            axes.push_back({a[0], a[1], a[2]});
        }
    }

    bool first = true;
    NelderMeadResult best;
    for (const Vec3& n0 : axes) {
        const std::vector<double> x0 = {std::atan2(n0.y, n0.x), std::asin(std::min(1.0, std::max(-1.0, n0.z))), dot(center, n0)};
        const NelderMeadResult r = nelder_mead(cost, x0, 1e-5, 1e-7, 400);
        if (first || r.fun < best.fun) best = r;
        first = false;
    }
    Plane plane;
    plane.n = normal_of(best.x[0], best.x[1]);
    plane.d = best.x[2];
    plane.residual = best.fun;
    return plane;
}

Mesh mirror_completion(const Mesh& surf, const Mesh& filled, const Vec3& n, double d, double far, double near, std::uint64_t seed) {
    Rng rng(seed);
    const double h = median(surf.edge_lengths());
    Vec3 lo, hi;
    filled.bounds(lo, hi);
    const double side = std::max(hi.x - lo.x, std::max(hi.y - lo.y, hi.z - lo.z));

    Points VR(surf.V.size());
    for (std::size_t i = 0; i < VR.size(); i++) VR[i] = reflect(surf.V[i], n, d);
    std::vector<Tri> FR(surf.F.size());
    for (std::size_t f = 0; f < FR.size(); f++) FR[f] = {surf.F[f][2], surf.F[f][1], surf.F[f][0]};
    const int m = static_cast<int>(FR.size());
    Points centers(m);
    for (int f = 0; f < m; f++) centers[f] = (VR[FR[f][0]] + VR[FR[f][1]] + VR[FR[f][2]]) / 3.0;

    const KdTree input_tree(sample_surface(surf.V, surf.F, std::max(200000, 20 * surf.n_faces()), rng));
    const std::vector<double> to_input = input_tree.nearest_distances(centers);
    const KdTree fill_tree(sample_surface(filled.V, filled.F, std::max(200000, 20 * filled.n_faces()), rng));
    const std::vector<double> to_fill = fill_tree.nearest_distances(centers);
    std::vector<char> in_void(m), seeds(m);
    for (int f = 0; f < m; f++) {
        in_void[f] = to_input[f] > far * h;
        seeds[f] = in_void[f] && to_fill[f] < near * side;
    }

    // Components of the mirrored faces in the void, joined by edges.
    const Mesh mirrored(VR, FR);
    std::vector<int> parent(m);
    std::iota(parent.begin(), parent.end(), 0);
    auto root = [&](int a) {
        while (parent[a] != a) { parent[a] = parent[parent[a]]; a = parent[a]; }
        return a;
    };
    std::vector<int> last(mirrored.n_edges(), -1);
    for (int f = 0; f < m; f++) {
        if (!in_void[f]) continue;
        for (int e : mirrored.face_edges[f]) {
            if (last[e] >= 0) parent[root(f)] = root(last[e]);
            last[e] = f;
        }
    }
    std::vector<char> seeded(m, 0);
    for (int f = 0; f < m; f++)
        if (seeds[f]) seeded[root(f)] = 1;
    std::vector<Tri> keep;
    for (int f = 0; f < m; f++)
        if (in_void[f] && seeded[root(f)]) keep.push_back(FR[f]);
    return compact(VR, keep);
}

Reconstruction reconstruct_with_symmetry(const Mesh& surf, bool symmetry, double max_residual, double gamma, int quadrature,
                                         const std::vector<Vec3>* axes) {
    Reconstruction r;
    auto run_pass = [&](const Mesh& target, Points& method_V, std::vector<Tri>& method_F, Points& faired_V, std::vector<Tri>& faired_F,
                        FairingStats& stats) {
        const TetMesh mesh = conforming_tet_mesh(target);
        SegmentOptions options;
        options.gamma = gamma;
        options.quadrature = quadrature;
        options.heuristics = true;
        const Segmentation s = graphcut_segment(target, options, &mesh);
        method_V = s.V;
        method_F = s.F;
        fair_filled_patches(s.V, s.F, target, faired_V, faired_F, 2, 3, 3, &stats);
    };
    run_pass(surf, r.method_V, r.method_F, r.faired_V, r.faired_F, r.fairing);
    if (!symmetry || r.faired_F.empty()) return r;

    r.plane = symmetry_plane(surf, 4000, 0.8, 0, axes);
    r.has_plane = true;
    if (r.plane.residual > max_residual * median(surf.edge_lengths())) return r;

    r.mirrored = mirror_completion(surf, Mesh(r.faired_V, r.faired_F), r.plane.n, r.plane.d);
    if (r.mirrored.n_faces() == 0) return r;
    Points V = surf.V;
    V.insert(V.end(), r.mirrored.V.begin(), r.mirrored.V.end());
    std::vector<Tri> F = surf.F;
    const int offset = surf.n_vertices();
    for (const Tri& t : r.mirrored.F) F.push_back({t[0] + offset, t[1] + offset, t[2] + offset});
    Points unused_V;
    std::vector<Tri> unused_F;
    run_pass(Mesh(V, F), unused_V, unused_F, r.symmetry_V, r.symmetry_F, r.fairing_second);
    r.has_symmetry = true;
    return r;
}

}  // namespace gsd
