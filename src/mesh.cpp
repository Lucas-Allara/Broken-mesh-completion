#include "gsd/mesh.h"

#include "gsd/rng.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace gsd {

Mesh::Mesh(Points vertices, std::vector<Tri> faces) : V(std::move(vertices)), F(std::move(faces)) {
    const int nv = n_vertices();
    for (const Tri& t : F)
        for (int v : t)
            if (v < 0 || v >= nv) throw std::invalid_argument("F has out-of-range indices");

    // The three directed edges of each face, opposite its corners 0, 1 and 2.
    std::vector<std::uint64_t> keys(3 * F.size());
    for (std::size_t f = 0; f < F.size(); f++)
        for (int i = 0; i < 3; i++) keys[3 * f + i] = pair_key(F[f][(i + 1) % 3], F[f][(i + 2) % 3]);
    std::vector<std::uint64_t> unique_keys = keys;
    std::sort(unique_keys.begin(), unique_keys.end());
    unique_keys.erase(std::unique(unique_keys.begin(), unique_keys.end()), unique_keys.end());

    edges.resize(unique_keys.size());
    for (std::size_t e = 0; e < unique_keys.size(); e++)
        edges[e] = {static_cast<int>(unique_keys[e] >> 32), static_cast<int>(unique_keys[e] & 0xffffffffu)};
    face_edges.resize(F.size());
    face_edge_sign.resize(F.size());
    edge_face_count.assign(edges.size(), 0);
    for (std::size_t f = 0; f < F.size(); f++)
        for (int i = 0; i < 3; i++) {
            const int e = static_cast<int>(std::lower_bound(unique_keys.begin(), unique_keys.end(), keys[3 * f + i]) - unique_keys.begin());
            face_edges[f][i] = e;
            face_edge_sign[f][i] = F[f][(i + 1) % 3] < F[f][(i + 2) % 3] ? 1 : -1;
            edge_face_count[e]++;
        }
}

std::vector<int> Mesh::boundary_edges() const {
    std::vector<int> out;
    for (int e = 0; e < n_edges(); e++)
        if (edge_face_count[e] == 1) out.push_back(e);
    return out;
}

bool Mesh::is_edge_manifold() const {
    return std::all_of(edge_face_count.begin(), edge_face_count.end(), [](int c) { return c <= 2; });
}

bool Mesh::is_closed() const {
    return std::all_of(edge_face_count.begin(), edge_face_count.end(), [](int c) { return c == 2; });
}

Vec3 Mesh::face_normal_raw(int f) const {
    const Vec3 &a = V[F[f][0]], &b = V[F[f][1]], &c = V[F[f][2]];
    return cross(b - a, c - a);
}

std::vector<double> Mesh::face_areas() const {
    std::vector<double> out(F.size());
    for (int f = 0; f < n_faces(); f++) out[f] = 0.5 * norm(face_normal_raw(f));
    return out;
}

std::vector<Vec3> Mesh::face_normals() const {
    std::vector<Vec3> out(F.size());
    for (int f = 0; f < n_faces(); f++) {
        const Vec3 n = face_normal_raw(f);
        out[f] = n / std::max(norm(n), 1e-14);
    }
    return out;
}

double Mesh::total_area() const {
    const std::vector<double> a = face_areas();
    return a.empty() ? 0.0 : pairwise_sum_of(a);
}

std::vector<std::array<double, 3>> Mesh::corner_angles() const {
    std::vector<std::array<double, 3>> out(F.size());
    for (int f = 0; f < n_faces(); f++)
        for (int i = 0; i < 3; i++) {
            const Vec3 u = V[F[f][(i + 1) % 3]] - V[F[f][i]];
            const Vec3 v = V[F[f][(i + 2) % 3]] - V[F[f][i]];
            out[f][i] = std::atan2(norm(cross(u, v)), dot(u, v));
        }
    return out;
}

std::vector<double> Mesh::edge_lengths() const {
    std::vector<double> out(edges.size());
    for (int e = 0; e < n_edges(); e++) out[e] = norm(V[edges[e][1]] - V[edges[e][0]]);
    return out;
}

Mesh Mesh::without_faces(const std::vector<int>& drop) const {
    std::vector<char> keep(F.size(), 1);
    for (int f : drop) keep[f] = 0;
    std::vector<Tri> faces;
    for (int f = 0; f < n_faces(); f++)
        if (keep[f]) faces.push_back(F[f]);
    return Mesh(V, std::move(faces));
}

Mesh Mesh::split_edges(const std::vector<char>& to_split) const {
    std::vector<int> mid(edges.size(), -1);
    Points vertices = V;
    for (int e = 0; e < n_edges(); e++)
        if (to_split[e]) {
            mid[e] = static_cast<int>(vertices.size());
            vertices.push_back(0.5 * (V[edges[e][0]] + V[edges[e][1]]));
        }

    std::vector<int> count(F.size());
    for (int f = 0; f < n_faces(); f++)
        count[f] = (to_split[face_edges[f][0]] != 0) + (to_split[face_edges[f][1]] != 0) + (to_split[face_edges[f][2]] != 0);

    // The blocks go in a fixed order, that of the reference values of the tests: whole faces, then
    // the faces with one split edge (two triangles each, one kind after the other), with two and
    // with three.
    std::vector<Tri> faces;
    for (int f = 0; f < n_faces(); f++)
        if (count[f] == 0) faces.push_back(F[f]);

    // One split edge: corner 0 opposite the split edge.
    for (int piece = 0; piece < 2; piece++)
        for (int f = 0; f < n_faces(); f++) {
            if (count[f] != 1) continue;
            int k = 0;
            while (!to_split[face_edges[f][k]]) k++;
            const int p0 = F[f][k], p1 = F[f][(k + 1) % 3], p2 = F[f][(k + 2) % 3], m0 = mid[face_edges[f][k]];
            faces.push_back(piece == 0 ? Tri{p0, p1, m0} : Tri{p0, m0, p2});
        }
    // Two split edges: corner 0 opposite the edge that was not split.
    for (int piece = 0; piece < 3; piece++)
        for (int f = 0; f < n_faces(); f++) {
            if (count[f] != 2) continue;
            int k = 0;
            while (to_split[face_edges[f][k]]) k++;
            const int p0 = F[f][k], p1 = F[f][(k + 1) % 3], p2 = F[f][(k + 2) % 3];
            const int m1 = mid[face_edges[f][(k + 1) % 3]], m2 = mid[face_edges[f][(k + 2) % 3]];
            faces.push_back(piece == 0 ? Tri{p0, m2, m1} : (piece == 1 ? Tri{m2, p1, p2} : Tri{m2, p2, m1}));
        }
    for (int piece = 0; piece < 4; piece++)
        for (int f = 0; f < n_faces(); f++) {
            if (count[f] != 3) continue;
            const int a = F[f][0], b = F[f][1], c = F[f][2];
            const int ma = mid[face_edges[f][0]], mb = mid[face_edges[f][1]], mc = mid[face_edges[f][2]];
            const Tri options[4] = {{a, mc, mb}, {mc, b, ma}, {mb, ma, c}, {ma, mb, mc}};
            faces.push_back(options[piece]);
        }
    return Mesh(std::move(vertices), std::move(faces));
}

Mesh load_obj(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open " + path);
    Points vertices;
    std::vector<Tri> faces;
    std::string line;
    int number = 0;
    auto malformed = [&](const char* what) { return std::runtime_error(path + ":" + std::to_string(number) + ": malformed " + what); };
    while (std::getline(in, line)) {
        number++;
        if (line.rfind("v ", 0) == 0) {
            const char* s = line.c_str() + 2;
            Vec3 p;
            for (int a = 0; a < 3; a++) {
                char* end;
                p[a] = std::strtod(s, &end);
                if (end == s) throw malformed("vertex");
                s = end;
            }
            vertices.push_back(p);
        } else if (line.rfind("f ", 0) == 0) {
            // The tokens can be "v", "v/vt", "v//vn" or "v/vt/vn".
            std::istringstream ss(line.substr(2));
            std::string token;
            std::vector<int> idx;
            while (ss >> token) {
                char* end;
                const long i = std::strtol(token.c_str(), &end, 10);
                if (end == token.c_str() || i == 0) throw malformed("face");
                // OBJ counts from 1; negative indices count from the end.
                idx.push_back(i > 0 ? static_cast<int>(i - 1) : static_cast<int>(vertices.size()) + static_cast<int>(i));
            }
            if (idx.size() < 3) throw malformed("face");
            for (std::size_t k = 1; k + 1 < idx.size(); k++) faces.push_back({idx[0], idx[k], idx[k + 1]});
        }
    }
    if (faces.empty()) throw std::runtime_error(path + " has no faces");
    try {
        return Mesh(std::move(vertices), std::move(faces));
    } catch (const std::invalid_argument&) {
        throw std::runtime_error(path + ": a face refers to a vertex that does not exist");
    }
}

void save_obj(const Points& V, const std::vector<Tri>& F, const std::string& path) {
    const std::filesystem::path parent = std::filesystem::path(path).parent_path();
    if (!parent.empty()) std::filesystem::create_directories(parent);
    std::FILE* out = std::fopen(path.c_str(), "w");
    if (!out) throw std::runtime_error("cannot write " + path);
    for (const Vec3& v : V) std::fprintf(out, "v %.17g %.17g %.17g\n", v.x, v.y, v.z);
    for (const Tri& f : F) std::fprintf(out, "f %d %d %d\n", f[0] + 1, f[1] + 1, f[2] + 1);
    std::fclose(out);
}

void save_obj(const Mesh& mesh, const std::string& path) { save_obj(mesh.V, mesh.F, path); }

Mesh compact(const Points& V, const std::vector<Tri>& F) {
    std::vector<int> remap(V.size(), -1);
    for (const Tri& t : F)
        for (int v : t) remap[v] = 0;
    Points vertices;
    for (std::size_t v = 0; v < V.size(); v++)
        if (remap[v] == 0) {
            remap[v] = static_cast<int>(vertices.size());
            vertices.push_back(V[v]);
        }
    std::vector<Tri> faces(F.size());
    for (std::size_t f = 0; f < F.size(); f++) faces[f] = {remap[F[f][0]], remap[F[f][1]], remap[F[f][2]]};
    return Mesh(std::move(vertices), std::move(faces));
}

int connected_components(const Mesh& mesh) {
    std::vector<int> parent(mesh.n_vertices());
    std::iota(parent.begin(), parent.end(), 0);
    auto root = [&](int a) {
        while (parent[a] != a) { parent[a] = parent[parent[a]]; a = parent[a]; }
        return a;
    };
    for (const Edge& e : mesh.edges) parent[root(e[0])] = root(e[1]);
    int count = 0;
    for (int v = 0; v < mesh.n_vertices(); v++) count += root(v) == v;
    return count;
}

}  // namespace gsd
