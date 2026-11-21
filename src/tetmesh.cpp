#include "gsd/tetmesh.h"

#include <algorithm>
#include <stdexcept>

namespace gsd {
namespace {
double volume6(const Points& V, const Tet& t) {
    return dot(V[t[1]] - V[t[0]], cross(V[t[2]] - V[t[0]], V[t[3]] - V[t[0]]));
}
}  // namespace

TetMesh::TetMesh(Points vertices, std::vector<Tet> tets) : V(std::move(vertices)), T(std::move(tets)) {
    const int nv = n_vertices();
    for (Tet& t : T) {
        for (int v : t)
            if (v < 0 || v >= nv) throw std::invalid_argument("T has out-of-range indices");
        double vol = volume6(V, t) / 6.0;
        if (vol < 0) {
            std::swap(t[1], t[2]);
            vol = -vol;
        }
        if (vol < 1e-14) throw std::invalid_argument("there are degenerate tetrahedra (zero volume)");
    }

    std::vector<Tri> all(4 * T.size());
    for (std::size_t t = 0; t < T.size(); t++)
        for (int i = 0; i < 4; i++) all[4 * t + i] = sorted(Tri{T[t][(i + 1) % 4], T[t][(i + 2) % 4], T[t][(i + 3) % 4]});
    faces = all;
    std::sort(faces.begin(), faces.end());
    faces.erase(std::unique(faces.begin(), faces.end()), faces.end());
    tet_faces.resize(T.size());
    face_tet_count.assign(faces.size(), 0);
    for (std::size_t t = 0; t < T.size(); t++)
        for (int i = 0; i < 4; i++) {
            const int f = static_cast<int>(std::lower_bound(faces.begin(), faces.end(), all[4 * t + i]) - faces.begin());
            tet_faces[t][i] = f;
            face_tet_count[f]++;
        }
}

double TetMesh::signed_volume(int t) const { return volume6(V, T[t]) / 6.0; }

std::vector<double> TetMesh::face_areas() const {
    std::vector<double> out(faces.size());
    for (int f = 0; f < n_faces(); f++)
        out[f] = 0.5 * norm(cross(V[faces[f][1]] - V[faces[f][0]], V[faces[f][2]] - V[faces[f][0]]));
    return out;
}

std::vector<Tet> drop_degenerate_tets(const Points& V, const std::vector<Tet>& T, double tol) {
    std::vector<Tet> out;
    out.reserve(T.size());
    for (const Tet& t : T)
        if (std::fabs(volume6(V, t)) / 6.0 > tol) out.push_back(t);
    return out;
}

}  // namespace gsd
