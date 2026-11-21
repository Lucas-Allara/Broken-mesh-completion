// Generalized winding number (Jacobson, Kavan and Sorkine-Hornung, 2013).
//
// winding_number sums the solid angle of all the faces; WindingHierarchy is the hierarchical
// evaluation of Section 4.3, which gives the same result while summing far fewer triangles.
#pragma once
#include <memory>
#include <vector>

#include "gsd/mesh.h"

namespace gsd {

// Signed solid angle of the triangle abc seen from p (Van Oosterom and Strackee formula).
double solid_angle(const Vec3& a, const Vec3& b, const Vec3& c, const Vec3& p);

// w at each point, by the direct sum over all the faces.
std::vector<double> winding_number(const Mesh& mesh, const Points& points);

// Exact hierarchical evaluation (Algorithms 1 and 2 of the paper). If a point lies outside the box
// of a patch S, w_S(p) is minus the winding number of any surface that closes it, and the closure
// has far fewer faces than S. The tree splits the box in half along its longest side.
class WindingHierarchy {
public:
    explicit WindingHierarchy(const Mesh& mesh, int leaf_size = 100);
    ~WindingHierarchy();
    std::vector<double> operator()(const Points& points) const;
    double at(const Vec3& p) const;

private:
    struct Node;
    Points V_;
    int leaf_size_;
    double eps_;
    std::unique_ptr<Node> root_;
    std::unique_ptr<Node> build(std::vector<Tri> F, const Vec3& lo_part, const Vec3& hi_part) const;
    void evaluate(const Node& node, const Vec3& p, double& out, std::vector<double>& buffer) const;
    double direct(const std::vector<Tri>& F, const Vec3& p, std::vector<double>& buffer) const;
};

// Triangles that close the surface patch F: each edge that F traverses more times in one direction
// than in the other is joined to a fixed vertex with a triangle that traverses it backwards.
std::vector<Tri> closure(const std::vector<Tri>& F);

}  // namespace gsd
