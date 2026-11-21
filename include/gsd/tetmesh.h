// Tetrahedral mesh. Face i of a tetrahedron is the one opposite its vertex i.
#pragma once
#include <vector>

#include "gsd/types.h"

namespace gsd {

struct TetMesh {
    Points V;
    std::vector<Tet> T;

    std::vector<Tri> faces;                 // sorted triples, in lexicographic order
    std::vector<Tet> tet_faces;             // face opposite each vertex
    std::vector<int> face_tet_count;        // 1 on the boundary of the domain, 2 inside

    TetMesh() = default;
    // Flips the tetrahedra with negative volume; throws if one has zero volume.
    TetMesh(Points vertices, std::vector<Tet> tets);

    int n_vertices() const { return static_cast<int>(V.size()); }
    int n_tets() const { return static_cast<int>(T.size()); }
    int n_faces() const { return static_cast<int>(faces.size()); }

    double signed_volume(int t) const;
    std::vector<double> face_areas() const;
    Vec3 tet_centroid(int t) const { return (V[T[t][0]] + V[T[t][1]] + V[T[t][2]] + V[T[t][3]]) / 4.0; }
};

// Removes the tetrahedra with nearly zero volume, which TetGen sometimes leaves.
std::vector<Tet> drop_degenerate_tets(const Points& V, const std::vector<Tet>& T, double tol = 1e-12);

}  // namespace gsd
