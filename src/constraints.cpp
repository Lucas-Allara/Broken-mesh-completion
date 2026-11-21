#include "gsd/constraints.h"

#include <algorithm>
#include <deque>
#include <functional>
#include <numeric>
#include <unordered_map>

#include "gsd/kdtree.h"

namespace gsd {

std::vector<int> input_faces(const Mesh& surf, const TetMesh& mesh) {
    std::vector<double> d;
    const std::vector<int> ids = KdTree(mesh.V).nearest_indices(surf.V, &d);
    Vec3 lo, hi;
    bounding_box(mesh.V, lo, hi);
    const double tol = 1e-12 * norm(hi - lo);
    std::unordered_map<Tri, int, TriHash> index;
    index.reserve(2 * mesh.faces.size());
    for (int f = 0; f < mesh.n_faces(); f++) index[mesh.faces[f]] = f;
    std::vector<int> out;
    for (const Tri& t : surf.F) {
        if (d[t[0]] > tol || d[t[1]] > tol || d[t[2]] > tol) continue;
        const auto it = index.find(sorted(Tri{ids[t[0]], ids[t[1]], ids[t[2]]}));
        if (it != index.end()) out.push_back(it->second);
    }
    return out;
}

namespace {

// Rows of a table grouped by key, each group in its original order.
struct Csr {
    std::vector<int> start, value;
    Csr() = default;
    Csr(const std::vector<int>& keys, const std::vector<int>& values, int n) : start(n + 1, 0), value(values.size()) {
        for (int k : keys) start[k + 1]++;
        for (int i = 0; i < n; i++) start[i + 1] += start[i];
        std::vector<int> fill(start.begin(), start.end() - 1);
        for (std::size_t i = 0; i < keys.size(); i++) value[fill[keys[i]]++] = values[i];
    }
    const int* begin(int k) const { return value.data() + start[k]; }
    const int* end(int k) const { return value.data() + start[k + 1]; }
};

// The label of each tetrahedron and what is needed to change it one at a time.
class Labeling {
public:
    const TetMesh& mesh;
    const std::vector<double>& w;
    std::vector<char> x;
    const std::vector<double>&u_inside, &u_outside, &v_weights;
    bool manifold = false;

    std::vector<std::array<int, 2>> face_tets;
    std::vector<int> n_inside;          // inside tetrahedra of each face: it is on the output if there is exactly one
    std::vector<char> is_input;
    std::vector<Edge> edges;
    std::vector<Tri> face_edges;
    Csr edge_faces, vertex_faces, vertex_tets, neighbors, neighbor_pair;
    std::vector<std::array<int, 6>> tet_edges;

    Labeling(const TetMesh& mesh_, const std::vector<double>& w_, const std::vector<char>& inside, const std::vector<double>& u_in,
             const std::vector<double>& u_out, const Adjacency& adj, const std::vector<double>& v, const std::vector<int>& inputs)
        : mesh(mesh_), w(w_), x(inside), u_inside(u_in), u_outside(u_out), v_weights(v) {
        const int m = mesh.n_tets(), nf = mesh.n_faces();
        face_tets.assign(nf, {-1, -1});
        n_inside.assign(nf, 0);
        for (int t = 0; t < m; t++)
            for (int f : mesh.tet_faces[t]) {
                (face_tets[f][0] < 0 ? face_tets[f][0] : face_tets[f][1]) = t;
                if (x[t]) n_inside[f]++;
            }
        is_input.assign(nf, 0);
        for (int f : inputs) is_input[f] = 1;

        // Edges of the faces: (0,1), (1,2) and (2,0) of each sorted triple.
        std::vector<std::uint64_t> keys(3 * static_cast<std::size_t>(nf));
        for (int f = 0; f < nf; f++)
            for (int j = 0; j < 3; j++) keys[3 * static_cast<std::size_t>(f) + j] = pair_key(mesh.faces[f][j], mesh.faces[f][(j + 1) % 3]);
        std::vector<std::uint64_t> unique_keys = keys;
        std::sort(unique_keys.begin(), unique_keys.end());
        unique_keys.erase(std::unique(unique_keys.begin(), unique_keys.end()), unique_keys.end());
        const int ne = static_cast<int>(unique_keys.size());
        edges.resize(ne);
        for (int e = 0; e < ne; e++) edges[e] = {static_cast<int>(unique_keys[e] >> 32), static_cast<int>(unique_keys[e] & 0xffffffffu)};
        face_edges.resize(nf);
        std::vector<int> flat_edges(3 * static_cast<std::size_t>(nf)), flat_faces(3 * static_cast<std::size_t>(nf)), flat_vertices(3 * static_cast<std::size_t>(nf));
        for (int f = 0; f < nf; f++)
            for (int j = 0; j < 3; j++) {
                const std::size_t i = 3 * static_cast<std::size_t>(f) + j;
                face_edges[f][j] = static_cast<int>(std::lower_bound(unique_keys.begin(), unique_keys.end(), keys[i]) - unique_keys.begin());
                flat_edges[i] = face_edges[f][j];
                flat_faces[i] = f;
                flat_vertices[i] = mesh.faces[f][j];
            }
        edge_faces = Csr(flat_edges, flat_faces, ne);
        vertex_faces = Csr(flat_vertices, flat_faces, mesh.n_vertices());

        std::vector<int> tet_vertices(4 * static_cast<std::size_t>(m)), tet_ids(4 * static_cast<std::size_t>(m));
        tet_edges.resize(m);
        for (int t = 0; t < m; t++) {
            int twelve[12];
            for (int i = 0; i < 4; i++) {
                tet_vertices[4 * static_cast<std::size_t>(t) + i] = mesh.T[t][i];
                tet_ids[4 * static_cast<std::size_t>(t) + i] = t;
                for (int j = 0; j < 3; j++) twelve[3 * i + j] = face_edges[mesh.tet_faces[t][i]][j];
            }
            std::sort(twelve, twelve + 12);
            for (int i = 0; i < 6; i++) tet_edges[t][i] = twelve[2 * i];
        }
        vertex_tets = Csr(tet_vertices, tet_ids, mesh.n_vertices());

        // Neighbors of each tetrahedron: first the pairs where it is tet_a, then where it is tet_b.
        const std::size_t np = adj.tet_a.size();
        std::vector<int> nkeys(2 * np), nvalues(2 * np), npairs(2 * np);
        for (std::size_t p = 0; p < np; p++) {
            nkeys[p] = adj.tet_a[p]; nvalues[p] = adj.tet_b[p]; npairs[p] = static_cast<int>(p);
            nkeys[np + p] = adj.tet_b[p]; nvalues[np + p] = adj.tet_a[p]; npairs[np + p] = static_cast<int>(p);
        }
        neighbors = Csr(nkeys, nvalues, m);
        neighbor_pair = Csr(nkeys, npairs, m);
    }

    void flip(int i) {
        x[i] = !x[i];
        for (int f : mesh.tet_faces[i]) n_inside[f] += x[i] ? 1 : -1;
    }

    bool edge_ok(int e) const {
        int count = 0;
        for (const int* f = edge_faces.begin(e); f != edge_faces.end(e); ++f) count += n_inside[*f] == 1;
        return count == 0 || count == 2;
    }

    // The output faces around v must form a single fan, joined by the edges that contain v.
    bool vertex_ok(int v) const {
        std::vector<int> faces;
        for (const int* f = vertex_faces.begin(v); f != vertex_faces.end(v); ++f)
            if (n_inside[*f] == 1) faces.push_back(*f);
        if (faces.empty()) return true;
        std::vector<int> parent(faces.size());
        std::iota(parent.begin(), parent.end(), 0);
        auto root = [&](int a) {
            while (parent[a] != a) { parent[a] = parent[parent[a]]; a = parent[a]; }
            return a;
        };
        std::unordered_map<int, int> first_face;
        for (std::size_t k = 0; k < faces.size(); k++)
            for (int e : face_edges[faces[k]]) {
                if (edges[e][0] != v && edges[e][1] != v) continue;
                const auto it = first_face.find(e);
                if (it != first_face.end()) parent[root(static_cast<int>(k))] = root(it->second);
                else first_face[e] = static_cast<int>(k);
            }
        int roots = 0;
        for (std::size_t k = 0; k < faces.size(); k++) roots += root(static_cast<int>(k)) == static_cast<int>(k);
        return roots == 1;
    }

    // Change of the energy if the label of i is flipped.
    double delta(int i) const {
        const bool current = x[i];
        // cost[label][i] with cost = (u_outside, u_inside).
        double sum = 0.0;
        const int* nb = neighbors.begin(i);
        const int* pair = neighbor_pair.begin(i);
        const int n = static_cast<int>(neighbors.end(i) - nb);
        // The products are summed as numpy does it: in order for a few terms.
        double terms[8];
        std::vector<double> many;
        double* t = terms;
        if (n > 8) { many.resize(n); t = many.data(); }
        for (int k = 0; k < n; k++) t[k] = v_weights[pair[k]] * ((x[nb[k]] != 0) == current ? 1.0 : -1.0);
        for (int k = 0; k < n; k++) sum += t[k];
        const double flipped = current ? u_outside[i] : u_inside[i];
        const double kept = current ? u_inside[i] : u_outside[i];
        return flipped - kept + sum;
    }

    // Flips i if it does not break any constraint that held.
    bool try_flip(int i) {
        if (x[i])
            for (int f : mesh.tet_faces[i])
                if (is_input[f] && n_inside[f] == 1) return false;
        if (!manifold) {
            flip(i);
            return true;
        }
        bool before[10], after[10];
        for (int k = 0; k < 6; k++) before[k] = edge_ok(tet_edges[i][k]);
        for (int k = 0; k < 4; k++) before[6 + k] = vertex_ok(mesh.T[i][k]);
        flip(i);
        for (int k = 0; k < 6; k++) after[k] = edge_ok(tet_edges[i][k]);
        for (int k = 0; k < 4; k++) after[6 + k] = vertex_ok(mesh.T[i][k]);
        for (int k = 0; k < 10; k++)
            if (before[k] && !after[k]) {
                flip(i);
                return false;
            }
        return true;
    }

    // Lowers the energy by flipping neighbors of the changed tetrahedra.
    void improve(const std::vector<int>& changed) {
        std::deque<int> queue;
        for (int i : changed)
            for (const int* j = neighbors.begin(i); j != neighbors.end(i); ++j) queue.push_back(*j);
        while (!queue.empty()) {
            const int j = queue.front();
            queue.pop_front();
            if (delta(j) < -1e-12 && try_flip(j))
                for (const int* k = neighbors.begin(j); k != neighbors.end(j); ++k) queue.push_back(*k);
        }
    }

    // Moves inside the outside tetrahedra of tets, from largest to smallest w, until satisfied().
    void move_inside(std::vector<int> tets, const std::function<bool()>& satisfied) {
        tets.erase(std::remove_if(tets.begin(), tets.end(), [&](int t) { return x[t] != 0; }), tets.end());
        std::stable_sort(tets.begin(), tets.end(), [&](int a, int b) { return w[a] > w[b]; });
        std::vector<int> changed;
        for (int i : tets) {
            if (satisfied()) break;
            flip(i);
            changed.push_back(i);
        }
        improve(changed);
    }
};

}  // namespace

std::vector<char> enforce_optional_constraints(const TetMesh& mesh, const std::vector<int>& inputs, const std::vector<double>& w,
                                               const std::vector<char>& inside, const std::vector<double>& u_inside,
                                               const std::vector<double>& u_outside, const Adjacency& adj,
                                               const std::vector<double>& v_weights, int max_rounds) {
    Labeling e(mesh, w, inside, u_inside, u_outside, adj, v_weights, inputs);

    for (int f : inputs)
        if (e.n_inside[f] == 0) {
            // Its neighboring tetrahedron with the largest w (the first one on ties).
            int i = e.face_tets[f][0];
            if (e.face_tets[f][1] >= 0 && w[e.face_tets[f][1]] > w[i]) i = e.face_tets[f][1];
            e.flip(i);
            e.improve({i});
        }

    e.manifold = true;
    const int ne = static_cast<int>(e.edges.size());
    for (int round = 0; round < max_rounds; round++) {
        std::vector<int> count(ne, 0);
        for (int f = 0; f < mesh.n_faces(); f++)
            if (e.n_inside[f] == 1)
                for (int a : e.face_edges[f]) count[a]++;
        std::vector<int> bad_edges;
        for (int a = 0; a < ne; a++)
            if (count[a] != 0 && count[a] != 2) bad_edges.push_back(a);
        for (int a : bad_edges)
            if (!e.edge_ok(a)) {
                std::vector<int> tets;
                for (const int* f = e.edge_faces.begin(a); f != e.edge_faces.end(a); ++f)
                    for (int t : e.face_tets[*f])
                        if (t >= 0) tets.push_back(t);
                std::sort(tets.begin(), tets.end());
                tets.erase(std::unique(tets.begin(), tets.end()), tets.end());
                e.move_inside(std::move(tets), [&e, a] { return e.edge_ok(a); });
            }
        std::vector<char> on_output(mesh.n_vertices(), 0);
        for (int f = 0; f < mesh.n_faces(); f++)
            if (e.n_inside[f] == 1)
                for (int v : mesh.faces[f]) on_output[v] = 1;
        std::vector<int> bad_vertices;
        for (int v = 0; v < mesh.n_vertices(); v++)
            if (on_output[v] && !e.vertex_ok(v)) bad_vertices.push_back(v);
        for (int v : bad_vertices)
            if (!e.vertex_ok(v)) {
                std::vector<int> tets(e.vertex_tets.begin(v), e.vertex_tets.end(v));
                e.move_inside(std::move(tets), [&e, v] { return e.vertex_ok(v); });
            }
        if (bad_edges.empty() && bad_vertices.empty()) break;
    }
    return e.x;
}

}  // namespace gsd
