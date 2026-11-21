// Tetrahedralization with every input triangle as a face (Section 5 of the paper), through TetGen.
//
// TetGen aborts the process if two triangles of the constraint intersect. That is why those
// triangles are detected beforehand and subdivided; the pieces that still intersect are left out
// of the constraint, although they still count for w.
#pragma once
#include <vector>

#include "gsd/mesh.h"
#include "gsd/tetmesh.h"

namespace gsd {

// Mask of the triangles that intersect some other one. Two non-coplanar triangles intersect if an
// edge of one crosses the other; overlapping coplanar ones are not detected.
std::vector<char> self_intersecting_faces(const Mesh& surf);

// Splits the marked faces into four without leaving hanging vertices: the neighbors with one
// split edge are bisected, and the ones left with two or more are split into four as well.
Mesh red_green_refine(const Mesh& mesh, std::vector<char> marked);

// The CDT of the convex hull of the input, without new points. ids[k] is the vertex of the result
// that corresponds to vertex k of the input. refine_intersections is how many times the
// intersecting faces are subdivided; if TetGen rejects the result, one fewer time is tried.
TetMesh conforming_tet_mesh(const Mesh& surf, std::vector<int>* ids = nullptr, int refine_intersections = 2);

}  // namespace gsd
