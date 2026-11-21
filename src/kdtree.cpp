#include "gsd/kdtree.h"

#include <algorithm>
#include <numeric>

namespace gsd {
namespace {
constexpr int kLeaf = 16;

double box_distance2(const Vec3& q, const Vec3& lo, const Vec3& hi) {
    double d2 = 0.0;
    for (int a = 0; a < 3; a++) {
        const double d = q[a] < lo[a] ? lo[a] - q[a] : (q[a] > hi[a] ? q[a] - hi[a] : 0.0);
        d2 += d * d;
    }
    return d2;
}
}  // namespace

KdTree::KdTree(const Points& points) : points_(points), order_(points.size()) {
    std::iota(order_.begin(), order_.end(), 0);
    if (!points_.empty()) {
        nodes_.reserve(2 * points_.size() / kLeaf + 2);
        build(0, static_cast<int>(points_.size()));
    }
}

int KdTree::build(int lo, int hi) {
    Node node{lo, hi, -1, -1, 0, 0.0, {INFINITY, INFINITY, INFINITY}, {-INFINITY, -INFINITY, -INFINITY}};
    for (int i = lo; i < hi; i++) {
        node.bmin = min(node.bmin, points_[order_[i]]);
        node.bmax = max(node.bmax, points_[order_[i]]);
    }
    const int id = static_cast<int>(nodes_.size());
    nodes_.push_back(node);
    if (hi - lo > kLeaf) {
        const Vec3 size = node.bmax - node.bmin;
        const int axis = size.x >= size.y && size.x >= size.z ? 0 : (size.y >= size.z ? 1 : 2);
        const int mid = (lo + hi) / 2;
        std::nth_element(order_.begin() + lo, order_.begin() + mid, order_.begin() + hi,
                         [&](int a, int b) { return points_[a][axis] < points_[b][axis]; });
        const double cut = points_[order_[mid]][axis];
        const int left = build(lo, mid);
        const int right = build(mid, hi);
        nodes_[id].left = left;
        nodes_[id].right = right;
        nodes_[id].axis = axis;
        nodes_[id].cut = cut;
    }
    return id;
}

void KdTree::nearest_rec(int n, const Vec3& q, double& best2, int& best) const {
    const Node& node = nodes_[n];
    if (node.left < 0) {
        for (int i = node.lo; i < node.hi; i++) {
            const int p = order_[i];
            const Vec3 d = points_[p] - q;
            const double d2 = d.x * d.x + d.y * d.y + d.z * d.z;
            if (d2 < best2 || (d2 == best2 && p < best)) { best2 = d2; best = p; }
        }
        return;
    }
    const int first = q[node.axis] < node.cut ? node.left : node.right;
    const int second = first == node.left ? node.right : node.left;
    if (box_distance2(q, nodes_[first].bmin, nodes_[first].bmax) <= best2) nearest_rec(first, q, best2, best);
    if (box_distance2(q, nodes_[second].bmin, nodes_[second].bmax) <= best2) nearest_rec(second, q, best2, best);
}

int KdTree::nearest(const Vec3& q, double* distance) const {
    double best2 = INFINITY;
    int best = -1;
    if (!nodes_.empty()) nearest_rec(0, q, best2, best);
    if (distance) *distance = std::sqrt(best2);
    return best;
}

std::vector<double> KdTree::nearest_distances(const Points& queries) const {
    std::vector<double> out(queries.size());
#pragma omp parallel for schedule(dynamic, 256)
    for (std::size_t i = 0; i < queries.size(); i++) nearest(queries[i], &out[i]);
    return out;
}

std::vector<int> KdTree::nearest_indices(const Points& queries, std::vector<double>* distances) const {
    std::vector<int> out(queries.size());
    if (distances) distances->resize(queries.size());
#pragma omp parallel for schedule(dynamic, 256)
    for (std::size_t i = 0; i < queries.size(); i++) out[i] = nearest(queries[i], distances ? &(*distances)[i] : nullptr);
    return out;
}

void KdTree::within_rec(int n, const Vec3& q, double r2, std::vector<int>& out) const {
    const Node& node = nodes_[n];
    if (box_distance2(q, node.bmin, node.bmax) > r2) return;
    if (node.left < 0) {
        for (int i = node.lo; i < node.hi; i++) {
            const Vec3 d = points_[order_[i]] - q;
            if (d.x * d.x + d.y * d.y + d.z * d.z <= r2) out.push_back(order_[i]);
        }
        return;
    }
    within_rec(node.left, q, r2, out);
    within_rec(node.right, q, r2, out);
}

void KdTree::within(const Vec3& q, double radius, std::vector<int>& out) const {
    if (!nodes_.empty()) within_rec(0, q, radius * radius, out);
}

}  // namespace gsd
