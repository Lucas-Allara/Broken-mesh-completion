// Static k-d tree over 3D points, for nearest neighbors and for the points within a radius.
#pragma once
#include <vector>

#include "gsd/types.h"

namespace gsd {

class KdTree {
public:
    explicit KdTree(const Points& points);
    // Index of the nearest point (the lowest index on ties) and its distance.
    int nearest(const Vec3& q, double* distance = nullptr) const;
    // Distance from each query to its nearest point, in parallel.
    std::vector<double> nearest_distances(const Points& queries) const;
    // Nearest point of each query, in parallel.
    std::vector<int> nearest_indices(const Points& queries, std::vector<double>* distances = nullptr) const;
    // Indices of the points at distance <= radius of q, in no particular order.
    void within(const Vec3& q, double radius, std::vector<int>& out) const;
    std::size_t size() const { return points_.size(); }

private:
    struct Node {
        int lo, hi;        // range of order_
        int left, right;   // children, -1 in a leaf
        int axis;
        double cut;
        Vec3 bmin, bmax;   // box of its points
    };
    Points points_;
    std::vector<int> order_;
    std::vector<Node> nodes_;
    int build(int lo, int hi);
    void nearest_rec(int node, const Vec3& q, double& best2, int& best) const;
    void within_rec(int node, const Vec3& q, double r2, std::vector<int>& out) const;
};

}  // namespace gsd
