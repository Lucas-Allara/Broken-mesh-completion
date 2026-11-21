// Triangle mesh with its connectivity.
//
// Convention: corner i of a face is opposite the edge F[f][i+1] -> F[f][i+2]. Each edge is stored
// once as (min, max), in lexicographic order, and face_edge_sign[f][i] is +1 if the face traverses
// it in that direction and -1 otherwise.
#pragma once
#include <string>
#include <vector>

#include "gsd/types.h"

namespace gsd {

struct Mesh {
    Points V;
    std::vector<Tri> F;

    std::vector<Edge> edges;
    std::vector<Tri> face_edges;                    // edge opposite each corner
    std::vector<std::array<signed char, 3>> face_edge_sign;
    std::vector<int> edge_face_count;               // 1 boundary, 2 interior, more: non-manifold

    Mesh() = default;
    Mesh(Points vertices, std::vector<Tri> faces);

    int n_vertices() const { return static_cast<int>(V.size()); }
    int n_faces() const { return static_cast<int>(F.size()); }
    int n_edges() const { return static_cast<int>(edges.size()); }

    std::vector<int> boundary_edges() const;
    bool is_edge_manifold() const;
    bool is_closed() const;
    Vec3 face_normal_raw(int f) const;              // its norm is twice the area
    std::vector<double> face_areas() const;
    std::vector<Vec3> face_normals() const;
    double total_area() const;
    std::vector<std::array<double, 3>> corner_angles() const;
    std::vector<double> edge_lengths() const;
    Vec3 face_center(int f) const { return (V[F[f][0]] + V[F[f][1]] + V[F[f][2]]) / 3.0; }
    void bounds(Vec3& lo, Vec3& hi) const { bounding_box(V, lo, hi); }

    Mesh without_faces(const std::vector<int>& drop) const;
    // Splits at their midpoint the edges marked in to_split and retriangulates each face according
    // to how many of its edges were split. The old vertices keep their index, the midpoints go at
    // the end, and the faces that were not split come first, in their order.
    Mesh split_edges(const std::vector<char>& to_split) const;
};

// Reads the vertices and faces of an OBJ. Polygons are fan-triangulated.
Mesh load_obj(const std::string& path);
void save_obj(const Mesh& mesh, const std::string& path);
void save_obj(const Points& V, const std::vector<Tri>& F, const std::string& path);

// Removes the vertices that no face uses, keeping the order of the others.
Mesh compact(const Points& V, const std::vector<Tri>& F);

// Number of connected components, through shared vertices.
int connected_components(const Mesh& mesh);

}  // namespace gsd
