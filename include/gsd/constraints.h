// Optional constraints of Section 5.2 of the paper.
//
// They cannot be imposed inside the graph cut, so the paper applies them with two heuristics on
// the labeling given by the cut. Every input face must be a face of some inside tetrahedron: if
// it is not, its neighboring tetrahedron with the largest w is moved inside. The output surface
// must be a manifold: at each edge or vertex where it is not, the tetrahedra that touch it are
// moved inside, from largest to smallest w, until it is. After each step the energy of Equation 8
// is improved by flipping neighbors of what changed, without breaking constraints that already held.
#pragma once
#include <vector>

#include "gsd/mesh.h"
#include "gsd/segmentation.h"
#include "gsd/tetmesh.h"

namespace gsd {

// Indices in mesh.faces of the triangles of surf that are faces of the tetrahedralization (with
// exactly equal vertices), in the order of the faces of surf.
std::vector<int> input_faces(const Mesh& surf, const TetMesh& mesh);

// Applies the two heuristics to the labeling inside and returns the new one.
std::vector<char> enforce_optional_constraints(const TetMesh& mesh, const std::vector<int>& inputs, const std::vector<double>& w,
                                               const std::vector<char>& inside, const std::vector<double>& u_inside,
                                               const std::vector<double>& u_outside, const Adjacency& adj,
                                               const std::vector<double>& v_weights, int max_rounds = 50);

}  // namespace gsd
