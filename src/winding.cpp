#include "gsd/winding.h"

#include <algorithm>
#include <cmath>
#include <map>

#include "gsd/rng.h"

namespace gsd {
namespace {
const double kInv4Pi = 1.0 / (4.0 * M_PI);
}

double solid_angle(const Vec3& va, const Vec3& vb, const Vec3& vc, const Vec3& p) {
    const Vec3 a = va - p, b = vb - p, c = vc - p;
    const double la = norm(a), lb = norm(b), lc = norm(c);
    const double num = dot(a, cross(b, c));
    const double den = la * lb * lc + dot(a, b) * lc + dot(b, c) * la + dot(c, a) * lb;
    return 2.0 * std::atan2(num, den);
}

std::vector<double> winding_number(const Mesh& mesh, const Points& points) {
    std::vector<double> out(points.size(), 0.0);
    const std::size_t m = mesh.F.size();
    if (m == 0) return out;
#pragma omp parallel
    {
        std::vector<double> buffer(m);
#pragma omp for schedule(dynamic, 16)
        for (std::size_t i = 0; i < points.size(); i++) {
            for (std::size_t f = 0; f < m; f++)
                buffer[f] = solid_angle(mesh.V[mesh.F[f][0]], mesh.V[mesh.F[f][1]], mesh.V[mesh.F[f][2]], points[i]);
            out[i] = pairwise_sum(buffer.data(), m) * kInv4Pi;
        }
    }
    return out;
}

std::vector<Tri> closure(const std::vector<Tri>& F) {
    // Net number of times that each edge is traversed from its lower to its higher vertex.
    std::map<std::uint64_t, int> count;
    for (const Tri& t : F)
        for (int i = 0; i < 3; i++) {
            const int a = t[i], b = t[(i + 1) % 3];
            count[pair_key(a, b)] += a < b ? 1 : -1;
        }
    std::vector<Tri> out;
    if (F.empty()) return out;
    const int k = F[0][0];
    for (const auto& [key, c] : count) {
        if (c == 0) continue;
        const int i = static_cast<int>(key >> 32), j = static_cast<int>(key & 0xffffffffu);
        const Tri tri = c > 0 ? Tri{j, i, k} : Tri{i, j, k};
        if (tri[0] == k || tri[1] == k) continue;  // degenerate
        for (int r = 0; r < std::abs(c); r++) out.push_back(tri);
    }
    return out;
}

struct WindingHierarchy::Node {
    Vec3 lo, hi;                 // box of the vertices of its faces
    std::vector<Tri> closure;
    std::vector<Tri> faces;      // only in the leaves
    std::unique_ptr<Node> left, right;
};

WindingHierarchy::WindingHierarchy(const Mesh& mesh, int leaf_size) : V_(mesh.V), leaf_size_(leaf_size) {
    Vec3 lo, hi;
    bounding_box(V_, lo, hi);
    eps_ = 1e-9 * norm(hi - lo);
    if (mesh.F.empty()) return;
    Vec3 flo{INFINITY, INFINITY, INFINITY}, fhi{-INFINITY, -INFINITY, -INFINITY};
    for (const Tri& t : mesh.F)
        for (int v : t) { flo = min(flo, V_[v]); fhi = max(fhi, V_[v]); }
    root_ = build(mesh.F, flo, fhi);
}

WindingHierarchy::~WindingHierarchy() = default;

std::unique_ptr<WindingHierarchy::Node> WindingHierarchy::build(std::vector<Tri> F, const Vec3& lo_part, const Vec3& hi_part) const {
    auto node = std::make_unique<Node>();
    // The "far away" test uses the actual box of the faces, not that of the partition, because a
    // face can stick out of the box of its child.
    node->lo = {INFINITY, INFINITY, INFINITY};
    node->hi = {-INFINITY, -INFINITY, -INFINITY};
    for (const Tri& t : F)
        for (int v : t) { node->lo = min(node->lo, V_[v]); node->hi = max(node->hi, V_[v]); }
    node->closure = closure(F);
    if (static_cast<int>(F.size()) < leaf_size_ || node->closure.size() >= F.size()) {
        node->faces = std::move(F);
        return node;
    }
    const Vec3 size = hi_part - lo_part;
    const int axis = size.x >= size.y && size.x >= size.z ? 0 : (size.y >= size.z ? 1 : 2);
    const double cut = 0.5 * (lo_part[axis] + hi_part[axis]);
    std::vector<Tri> left, right;
    for (const Tri& t : F) {
        const double c = (V_[t[0]][axis] + V_[t[1]][axis] + V_[t[2]][axis]) / 3.0;
        (c < cut ? left : right).push_back(t);
    }
    if (left.empty() || right.empty()) {
        node->faces = std::move(F);
        return node;
    }
    Vec3 hi_left = hi_part, lo_right = lo_part;
    hi_left[axis] = cut;
    lo_right[axis] = cut;
    node->left = build(std::move(left), lo_part, hi_left);
    node->right = build(std::move(right), lo_right, hi_part);
    return node;
}

double WindingHierarchy::direct(const std::vector<Tri>& F, const Vec3& p, std::vector<double>& buffer) const {
    if (F.empty()) return 0.0;
    if (buffer.size() < F.size()) buffer.resize(F.size());
    for (std::size_t f = 0; f < F.size(); f++) buffer[f] = solid_angle(V_[F[f][0]], V_[F[f][1]], V_[F[f][2]], p);
    return pairwise_sum(buffer.data(), F.size()) * kInv4Pi;
}

void WindingHierarchy::evaluate(const Node& node, const Vec3& p, double& out, std::vector<double>& buffer) const {
    if (!node.left) {
        out += direct(node.faces, p, buffer);
        return;
    }
    bool outside = false;
    for (int a = 0; a < 3; a++) outside |= p[a] < node.lo[a] - eps_ || p[a] > node.hi[a] + eps_;
    if (outside) {
        out -= direct(node.closure, p, buffer);
        return;
    }
    // The second child first, the order of the sums behind the reference values of the tests.
    evaluate(*node.right, p, out, buffer);
    evaluate(*node.left, p, out, buffer);
}

double WindingHierarchy::at(const Vec3& p) const {
    double out = 0.0;
    std::vector<double> buffer;
    if (root_) evaluate(*root_, p, out, buffer);
    return out;
}

std::vector<double> WindingHierarchy::operator()(const Points& points) const {
    std::vector<double> out(points.size(), 0.0);
    if (!root_) return out;
#pragma omp parallel
    {
        std::vector<double> buffer;
#pragma omp for schedule(dynamic, 64)
        for (std::size_t i = 0; i < points.size(); i++) evaluate(*root_, points[i], out[i], buffer);
    }
    return out;
}

}  // namespace gsd
