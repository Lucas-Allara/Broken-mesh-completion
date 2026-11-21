#include "gsd/segmentation.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <unordered_map>

#include "gsd/conforming.h"
#include "gsd/constraints.h"
#include "gsd/kdtree.h"
#include "gsd/maxflow.h"
#include "gsd/rng.h"
#include "gsd/winding.h"

namespace gsd {

Adjacency tet_adjacency(const TetMesh& mesh) {
    // The two tetrahedra of each interior face, the one with the lower index first.
    std::vector<std::array<int, 2>> face_tets(mesh.n_faces(), {-1, -1});
    for (int t = 0; t < mesh.n_tets(); t++)
        for (int f : mesh.tet_faces[t]) (face_tets[f][0] < 0 ? face_tets[f][0] : face_tets[f][1]) = t;
    const std::vector<double> areas = mesh.face_areas();
    Adjacency adj;
    for (int f = 0; f < mesh.n_faces(); f++)
        if (mesh.face_tet_count[f] == 2) {
            adj.tet_a.push_back(face_tets[f][0]);
            adj.tet_b.push_back(face_tets[f][1]);
            adj.area.push_back(areas[f]);
        }
    return adj;
}

std::vector<double> element_winding_number(const Mesh& surf, const TetMesh& mesh, int quadrature, bool hierarchical) {
    auto evaluate = [&](const Points& P) { return hierarchical ? WindingHierarchy(surf)(P) : winding_number(surf, P); };
    const int m = mesh.n_tets();
    if (quadrature == 1) {
        Points P(m);
        for (int t = 0; t < m; t++) P[t] = mesh.tet_centroid(t);
        return evaluate(P);
    }
    if (quadrature != 4) throw std::invalid_argument("quadrature must be 1 or 4");
    // Symmetric 4-point rule on the tetrahedron (exact up to degree 2), in barycentric coordinates.
    const double A = (5.0 + 3.0 * std::sqrt(5.0)) / 20.0, B = (5.0 - std::sqrt(5.0)) / 20.0;
    Points P(4 * static_cast<std::size_t>(m));
    for (int q = 0; q < 4; q++)
        for (int t = 0; t < m; t++) {
            Vec3 p{0, 0, 0};
            for (int k = 0; k < 4; k++) p += mesh.V[mesh.T[t][k]] * (k == q ? A : B);
            P[static_cast<std::size_t>(q) * m + t] = p;
        }
    const std::vector<double> w4 = evaluate(P);
    std::vector<double> w(m);
    for (int t = 0; t < m; t++) w[t] = (((w4[t] + w4[m + t]) + w4[2 * static_cast<std::size_t>(m) + t]) + w4[3 * static_cast<std::size_t>(m) + t]) / 4.0;
    return w;
}

void data_term(const std::vector<double>& w, std::vector<double>& u_inside, std::vector<double>& u_outside) {
    u_inside.resize(w.size());
    u_outside.resize(w.size());
    for (std::size_t i = 0; i < w.size(); i++) {
        u_inside[i] = std::max(1.0 - w[i], 0.0);
        u_outside[i] = std::max(w[i], 0.0);
    }
}

std::vector<double> smoothness_weights(const std::vector<double>& w, const Adjacency& adj, double sigma) {
    std::vector<double> out(adj.tet_a.size());
    for (std::size_t k = 0; k < out.size(); k++) {
        const double d = w[adj.tet_a[k]] - w[adj.tet_b[k]];
        out[k] = adj.area[k] * std::exp(-(d * d) / (2.0 * (sigma * sigma)));
    }
    return out;
}

double percentile_linear(std::vector<double> values, double percentile) {
    const std::size_t n = values.size();
    if (n == 0) throw std::invalid_argument("percentile of an empty array");
    std::sort(values.begin(), values.end());
    const double virtual_index = (static_cast<double>(n) - 1.0) * (percentile / 100.0);
    std::size_t previous = static_cast<std::size_t>(std::floor(virtual_index));
    previous = std::min(previous, n - 1);
    const std::size_t next = std::min(previous + 1, n - 1);
    const double t = virtual_index - static_cast<double>(previous);
    const double a = values[previous], b = values[next], diff = b - a;
    return t >= 0.5 ? b - diff * (1.0 - t) : a + diff * t;
}

double estimate_sigma(const std::vector<double>& w, const Adjacency& adj, double percentile) {
    if (adj.tet_a.empty()) throw std::invalid_argument("the tetrahedralization has no pair of neighboring tetrahedra");
    std::vector<double> jumps(adj.tet_a.size());
    for (std::size_t k = 0; k < jumps.size(); k++) jumps[k] = std::fabs(w[adj.tet_a[k]] - w[adj.tet_b[k]]);
    return std::max(percentile_linear(std::move(jumps), percentile), 1e-6);
}

void apply_hard_constraints(std::vector<double>& u_inside, std::vector<double>& u_outside, const std::vector<double>& v_weights,
                            const std::vector<char>& force_inside, const std::vector<char>& force_outside) {
    if (force_inside.empty() && force_outside.empty()) return;
    const std::size_t m = u_inside.size();
    for (std::size_t i = 0; i < m; i++)
        if (!force_inside.empty() && !force_outside.empty() && force_inside[i] && force_outside[i])
            throw std::invalid_argument("a tetrahedron cannot be forced both inside and outside");
    // Larger than the sum of all the costs of the graph, so no minimum cut violates a constraint.
    const double infinity = 1.0 + pairwise_sum_of(u_inside) + pairwise_sum_of(u_outside) + 2.0 * pairwise_sum_of(v_weights);
    for (std::size_t i = 0; i < m; i++) {
        if (!force_inside.empty() && force_inside[i]) { u_outside[i] = infinity; u_inside[i] = 0.0; }
        if (!force_outside.empty() && force_outside[i]) { u_inside[i] = infinity; u_outside[i] = 0.0; }
    }
}

void hard_constraints_from_surface(const Mesh& surf, const TetMesh& mesh, std::vector<char>& force_inside, std::vector<char>& force_outside) {
    std::vector<double> d;
    const std::vector<int> ids = KdTree(mesh.V).nearest_indices(surf.V, &d);
    Vec3 lo, hi;
    bounding_box(mesh.V, lo, hi);
    const double tol = 1e-12 * norm(hi - lo);
    std::vector<std::array<int, 2>> face_tets(mesh.n_faces(), {-1, -1});
    for (int t = 0; t < mesh.n_tets(); t++)
        for (int f : mesh.tet_faces[t]) (face_tets[f][0] < 0 ? face_tets[f][0] : face_tets[f][1]) = t;
    std::unordered_map<Tri, int, TriHash> index;
    for (int f = 0; f < mesh.n_faces(); f++)
        if (mesh.face_tet_count[f] == 2) index[mesh.faces[f]] = f;

    const std::vector<Vec3> normals = surf.face_normals();
    force_inside.assign(mesh.n_tets(), 0);
    force_outside.assign(mesh.n_tets(), 0);
    for (int k = 0; k < surf.n_faces(); k++) {
        const Tri& tri = surf.F[k];
        if (d[tri[0]] > tol || d[tri[1]] > tol || d[tri[2]] > tol) continue;
        const auto it = index.find(sorted(Tri{ids[tri[0]], ids[tri[1]], ids[tri[2]]}));
        if (it == index.end()) continue;
        const int f = it->second, a = face_tets[f][0], b = face_tets[f][1];
        const Vec3 face_center = (mesh.V[mesh.faces[f][0]] + mesh.V[mesh.faces[f][1]] + mesh.V[mesh.faces[f][2]]) / 3.0;
        const bool a_outside = dot(normals[k], mesh.tet_centroid(a) - face_center) > 0;
        force_inside[a_outside ? b : a] = 1;
        force_outside[a_outside ? a : b] = 1;
    }
}

std::vector<char> min_cut_labels(const std::vector<double>& u_inside, const std::vector<double>& u_outside, const Adjacency& adj,
                                 const std::vector<double>& v_weights, double scale) {
    // Source = inside, sink = outside; S -> i costs u_outside(i), i -> T costs u_inside(i) and each
    // pair of neighbors carries an edge in each direction. The capacities are scaled to integers.
    const int m = static_cast<int>(u_inside.size());
    double peak = 0.0;
    for (double c : u_inside) peak = std::max(peak, c);
    for (double c : u_outside) peak = std::max(peak, c);
    for (double c : v_weights) peak = std::max(peak, c);
    if (!(peak > 0)) peak = 1.0;
    const double k = scale / peak;
    auto capacity = [&](double c) { return static_cast<std::int64_t>(std::nearbyint(c * k)); };

    const int S = m, T = m + 1;
    MaxFlow flow(m + 2);
    for (int i = 0; i < m; i++) {
        flow.add_edge(S, i, capacity(u_outside[i]));
        flow.add_edge(i, T, capacity(u_inside[i]));
    }
    for (std::size_t p = 0; p < v_weights.size(); p++) {
        const std::int64_t c = capacity(v_weights[p]);
        flow.add_edge(adj.tet_a[p], adj.tet_b[p], c, c);
    }
    flow.run(S, T);
    std::vector<char> side = flow.source_side(S);
    side.resize(m);
    return side;
}

double energy(const std::vector<char>& inside, const std::vector<double>& u_inside, const std::vector<double>& u_outside,
              const Adjacency& adj, const std::vector<double>& v_weights) {
    double e = 0.0;
    for (std::size_t i = 0; i < inside.size(); i++) e += inside[i] ? u_inside[i] : u_outside[i];
    for (std::size_t p = 0; p < v_weights.size(); p++)
        if (inside[adj.tet_a[p]] != inside[adj.tet_b[p]]) e += v_weights[p];
    return e;
}

void extract_label_boundary(const TetMesh& mesh, const std::vector<char>& inside, Points& V, std::vector<Tri>& F) {
    std::vector<int> n_inside(mesh.n_faces(), 0);
    for (int t = 0; t < mesh.n_tets(); t++)
        if (inside[t])
            for (int f : mesh.tet_faces[t]) n_inside[f]++;
    std::vector<Tri> faces;
    for (int t = 0; t < mesh.n_tets(); t++) {
        if (!inside[t]) continue;
        const Vec3 tet_center = mesh.tet_centroid(t);
        for (int f : mesh.tet_faces[t]) {
            if (n_inside[f] != 1) continue;
            Tri face = mesh.faces[f];
            const Vec3 face_center = (mesh.V[face[0]] + mesh.V[face[1]] + mesh.V[face[2]]) / 3.0;
            const Vec3 normal = cross(mesh.V[face[1]] - mesh.V[face[0]], mesh.V[face[2]] - mesh.V[face[0]]);
            if (!(dot(normal, face_center - tet_center) > 0)) std::swap(face[1], face[2]);
            faces.push_back(face);
        }
    }
    const Mesh out = compact(mesh.V, faces);
    V = out.V;
    F = out.F;
}

Segmentation graphcut_segment(const Mesh& surf, const SegmentOptions& options, const TetMesh* given) {
    Segmentation r;
    r.mesh = given ? *given : conforming_tet_mesh(surf);
    const TetMesh& mesh = r.mesh;

    std::vector<char> force_inside, force_outside;
    if (options.hard_constraints) {
        std::vector<char> unused;
        hard_constraints_from_surface(surf, mesh, force_inside, unused);
    }

    r.w = element_winding_number(surf, mesh, options.quadrature);
    const Adjacency adj = tet_adjacency(mesh);
    std::vector<double> u_inside, u_outside;
    data_term(r.w, u_inside, u_outside);
    r.sigma = options.sigma < 0 ? estimate_sigma(r.w, adj) : options.sigma;
    std::vector<double> v_weights = smoothness_weights(r.w, adj, r.sigma);
    for (double& v : v_weights) v *= options.gamma;
    apply_hard_constraints(u_inside, u_outside, v_weights, force_inside, force_outside);

    r.inside = min_cut_labels(u_inside, u_outside, adj, v_weights);
    if (options.heuristics)
        r.inside = enforce_optional_constraints(mesh, input_faces(surf, mesh), r.w, r.inside, u_inside, u_outside, adj, v_weights);
    extract_label_boundary(mesh, r.inside, r.V, r.F);
    return r;
}

}  // namespace gsd
