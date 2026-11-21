// Segmentation by graph cut (Section 5 of the paper).
//
// Each tetrahedron of a conforming tetrahedralization is labeled inside or outside by minimizing
// the energy of Equation 8: a data term that comes from w and a smoothness term between
// neighbors. The output is the set of faces that separate the two labels.
#pragma once
#include <vector>

#include "gsd/mesh.h"
#include "gsd/tetmesh.h"

namespace gsd {

// Pairs of tetrahedra that share a face and the area of that face: the N(i) and the a_ij of
// Equation 10.
struct Adjacency {
    std::vector<int> tet_a, tet_b;
    std::vector<double> area;
};
Adjacency tet_adjacency(const TetMesh& mesh);

// w averaged over each tetrahedron, at the barycenter (quadrature 1) or with the 4-point rule.
std::vector<double> element_winding_number(const Mesh& surf, const TetMesh& mesh, int quadrature = 4, bool hierarchical = true);

// Equation 9: cost of labeling each element inside and outside.
void data_term(const std::vector<double>& w, std::vector<double>& u_inside, std::vector<double>& u_outside);
// Equation 10: cost of separating two neighbors, small where w already jumps between them.
std::vector<double> smoothness_weights(const std::vector<double>& w, const Adjacency& adj, double sigma);
// sigma of Equation 10, which the paper does not fix: a high percentile of |w_i - w_j| between
// neighbors (the median gives a tiny sigma, since almost all pairs are far from the surface).
double estimate_sigma(const std::vector<double>& w, const Adjacency& adj, double percentile = 85.0);
// numpy.percentile with linear interpolation.
double percentile_linear(std::vector<double> values, double percentile);

// Fixes the label of some tetrahedra by setting an infinite cost on the opposite side.
void apply_hard_constraints(std::vector<double>& u_inside, std::vector<double>& u_outside, const std::vector<double>& v_weights,
                            const std::vector<char>& force_inside, const std::vector<char>& force_outside);
// For each triangle of surf that is a face of mesh, marks as inside the tetrahedron on the side
// opposite the normal and as outside the one on the side of the normal.
void hard_constraints_from_surface(const Mesh& surf, const TetMesh& mesh, std::vector<char>& force_inside, std::vector<char>& force_outside);

// Minimizes Equation 8 with a minimum s-t cut. Returns 1 for the elements that end up inside.
std::vector<char> min_cut_labels(const std::vector<double>& u_inside, const std::vector<double>& u_outside, const Adjacency& adj,
                                 const std::vector<double>& v_weights, double scale = 1048576.0);
// Value of Equation 8 for a labeling.
double energy(const std::vector<char>& inside, const std::vector<double>& u_inside, const std::vector<double>& u_outside,
              const Adjacency& adj, const std::vector<double>& v_weights);

// Faces with exactly one inside tetrahedron, oriented outward and with the vertices compacted.
void extract_label_boundary(const TetMesh& mesh, const std::vector<char>& inside, Points& V, std::vector<Tri>& F);

struct SegmentOptions {
    double gamma = 1.0;
    double sigma = -1.0;            // negative: estimated
    int quadrature = 4;
    bool hard_constraints = false;  // those of hard_constraints_from_surface, inside only
    bool heuristics = false;        // the optional constraints of Section 5.2 after the cut
};

struct Segmentation {
    std::vector<double> w;
    std::vector<char> inside;
    Points V;
    std::vector<Tri> F;
    TetMesh mesh;
    double sigma = 0.0;
};

// The complete method. With gamma = 0 it gives the same result as the threshold w > 0.5.
Segmentation graphcut_segment(const Mesh& surf, const SegmentOptions& options = {}, const TetMesh* mesh = nullptr);

}  // namespace gsd
