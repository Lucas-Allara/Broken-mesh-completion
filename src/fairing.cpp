#include "gsd/fairing.h"

#include <Eigen/Sparse>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <set>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

#include "gsd/kdtree.h"

namespace gsd {

std::vector<char> on_input(const Points& P, const Mesh& surf, double tol) {
    const int m = surf.n_faces();
    Points centers(m);
    double radius = 0.0;
    for (int f = 0; f < m; f++) {
        centers[f] = surf.face_center(f);
        for (int v : surf.F[f]) radius = std::max(radius, norm(surf.V[v] - centers[f]));
    }
    const KdTree tree(centers);
    std::vector<char> on(P.size(), 0);
#pragma omp parallel
    {
        std::vector<int> near;
#pragma omp for schedule(dynamic, 256)
        for (std::size_t i = 0; i < P.size(); i++) {
            near.clear();
            tree.within(P[i], radius + tol, near);
            const Vec3& q = P[i];
            for (int j : near) {
                const Vec3 &a = surf.V[surf.F[j][0]], &b = surf.V[surf.F[j][1]], &c = surf.V[surf.F[j][2]];
                Vec3 n = cross(b - a, c - a);
                const double area2 = norm(n);
                if (!(area2 > 0)) continue;
                n = n / area2;
                if (std::fabs(dot(q - a, n)) > tol) continue;
                const double tol_b = tol * (norm(b - a) + norm(c - a));
                if (dot(cross(b - a, q - a), n) >= -tol_b && dot(cross(c - b, q - b), n) >= -tol_b && dot(cross(a - c, q - c), n) >= -tol_b) {
                    on[i] = 1;
                    break;
                }
            }
        }
    }
    return on;
}

namespace {

// How far a cap may stick out of the box of its anchor vertices, as a fraction of the diagonal of
// that box, before its order is lowered.
constexpr double kMargin = 0.15;

using Sparse = Eigen::SparseMatrix<double>;

// Solves L^degree x = 0 at the free vertices, with L the uniform Laplacian and the fixed ones as
// boundary. With degree 2 or 3 a cap can shoot far away and leave a spike: each cap that leaves
// the box of its anchors (plus the margin) is solved again with one degree less; with degree 1
// that does not happen (maximum principle). The free pieces that do not touch any fixed vertex
// stay where they are. stats, when given, receives the number of caps and the degree each ended with.
Points smooth(const Points& V, const std::vector<Tri>& F, const std::vector<char>& fixed_in, int degree, FairingStats* stats = nullptr) {
    const int n = static_cast<int>(V.size());
    const Mesh mesh(V, F);
    std::vector<std::vector<int>> adjacent(n);
    for (const Edge& e : mesh.edges) {
        adjacent[e[0]].push_back(e[1]);
        adjacent[e[1]].push_back(e[0]);
    }
    std::vector<char> fixed = fixed_in;

    // Components of the free vertices, numbered by their lowest vertex.
    auto components = [&](std::vector<int>& comp) {
        comp.assign(n, -1);
        int count = 0;
        std::vector<int> stack;
        for (int s = 0; s < n; s++) {
            if (fixed[s] || comp[s] >= 0) continue;
            comp[s] = count;
            stack.push_back(s);
            while (!stack.empty()) {
                const int u = stack.back();
                stack.pop_back();
                for (int v : adjacent[u])
                    if (!fixed[v] && comp[v] < 0) { comp[v] = count; stack.push_back(v); }
            }
            count++;
        }
        return count;
    };
    std::vector<int> comp;
    int n_comp = components(comp);
    std::vector<char> touches(n_comp, 0);
    for (int v = 0; v < n; v++)
        if (!fixed[v])
            for (int u : adjacent[v])
                if (fixed[u]) { touches[comp[v]] = 1; break; }
    for (int v = 0; v < n; v++)
        if (!fixed[v] && !touches[comp[v]]) fixed[v] = 1;
    n_comp = components(comp);
    if (stats) *stats = FairingStats();
    if (n_comp == 0) return V;

    // Box of the anchors of each cap: the fixed vertices next to it.
    std::vector<Vec3> box_lo(n_comp, {INFINITY, INFINITY, INFINITY}), box_hi(n_comp, {-INFINITY, -INFINITY, -INFINITY});
    for (int v = 0; v < n; v++)
        if (!fixed[v])
            for (int u : adjacent[v])
                if (fixed[u]) {
                    box_lo[comp[v]] = min(box_lo[comp[v]], V[u]);
                    box_hi[comp[v]] = max(box_hi[comp[v]], V[u]);
                }
    for (int k = 0; k < n_comp; k++) {
        const double margin = kMargin * norm(box_hi[k] - box_lo[k]);
        const Vec3 pad{margin, margin, margin};
        box_lo[k] = box_lo[k] - pad;
        box_hi[k] = box_hi[k] + pad;
    }

    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(n + 2 * mesh.edges.size());
    for (int v = 0; v < n; v++) triplets.emplace_back(v, v, static_cast<double>(adjacent[v].size()));
    for (const Edge& e : mesh.edges) {
        triplets.emplace_back(e[0], e[1], -1.0);
        triplets.emplace_back(e[1], e[0], -1.0);
    }
    Sparse L(n, n);
    L.setFromTriplets(triplets.begin(), triplets.end());

    Points X = V;
    std::vector<char> pending(n_comp, 1);
    std::vector<int> solved_with(n_comp, degree);
    for (int o = degree; o >= 1; o--) {
        // The caps already solved do not move.
        std::vector<int> index(n, -1);
        int n_move = 0;
        for (int v = 0; v < n; v++)
            if (!fixed[v] && pending[comp[v]]) index[v] = n_move++;
        Sparse M = L;
        for (int k = 1; k < o; k++) M = (L * M).pruned(0.0);

        std::vector<Eigen::Triplet<double>> system;
        Eigen::MatrixXd rhs = Eigen::MatrixXd::Zero(n_move, 3);
        for (int col = 0; col < M.outerSize(); col++)
            for (Sparse::InnerIterator it(M, col); it; ++it) {
                const int row = static_cast<int>(it.row());
                if (index[row] < 0) continue;
                if (index[col] >= 0) system.emplace_back(index[row], index[col], it.value());
                else
                    for (int a = 0; a < 3; a++) rhs(index[row], a) -= it.value() * X[col][a];
            }
        Sparse A(n_move, n_move);
        A.setFromTriplets(system.begin(), system.end());
        Eigen::SparseLU<Sparse> solver;
        solver.compute(A);
        if (solver.info() != Eigen::Success) throw std::runtime_error("fairing: the system could not be factorized");
        const Eigen::MatrixXd solution = solver.solve(rhs);
        for (int v = 0; v < n; v++)
            if (index[v] >= 0) X[v] = {solution(index[v], 0), solution(index[v], 1), solution(index[v], 2)};
        if (o == 1) break;

        std::vector<char> out_of_box(n_comp, 0);
        for (int v = 0; v < n; v++) {
            if (fixed[v] || !pending[comp[v]]) continue;
            const int k = comp[v];
            for (int a = 0; a < 3; a++)
                if (X[v][a] < box_lo[k][a] || X[v][a] > box_hi[k][a]) out_of_box[k] = 1;
        }
        bool any = false;
        for (int k = 0; k < n_comp; k++) {
            if (pending[k]) pending[k] = out_of_box[k];
            if (pending[k]) solved_with[k] = o - 1;
            any |= pending[k] != 0;
        }
        if (!any) break;
    }
    if (stats) {
        stats->caps = n_comp;
        for (int k = 0; k < n_comp; k++)
            if (solved_with[k] >= 1 && solved_with[k] <= 3) stats->by_order[solved_with[k]]++;
    }
    return X;
}

// Splits at their midpoint all the edges of the caps.
void split_caps(Points& V, std::vector<Tri>& F, std::vector<char>& is_input, std::vector<char>& fixed) {
    const Mesh mesh(V, F);
    std::vector<char> to_split(mesh.n_edges(), 1);
    for (int f = 0; f < mesh.n_faces(); f++)
        if (is_input[f])
            for (int e : mesh.face_edges[f]) to_split[e] = 0;
    const Mesh fine = mesh.split_edges(to_split);
    fixed.resize(fine.n_vertices(), 0);
    // split_edges puts first, in order, the faces that were not split, and all the input faces
    // are among them.
    std::vector<char> flags;
    for (int f = 0; f < mesh.n_faces(); f++)
        if (!to_split[mesh.face_edges[f][0]] && !to_split[mesh.face_edges[f][1]] && !to_split[mesh.face_edges[f][2]]) flags.push_back(is_input[f]);
    flags.resize(fine.n_faces(), 0);
    is_input = std::move(flags);
    V = fine.V;
    F = fine.F;
}

double angle_at(const Vec3& p, const Vec3& q, const Vec3& r) {
    const Vec3 u = q - p, v = r - p;
    return std::atan2(norm(cross(u, v)), dot(u, v));
}

// Flips the edges of the caps that are not locally Delaunay. The flip is not done if the new edge
// already exists or if some new triangle ends up inverted.
void delaunay_flips(const Points& V, std::vector<Tri>& F, const std::vector<char>& is_input, int max_passes = 50) {
    for (int pass = 0; pass < max_passes; pass++) {
        const Mesh mesh(V, F);
        std::unordered_set<std::uint64_t> existing;
        existing.reserve(2 * mesh.edges.size());
        for (const Edge& e : mesh.edges) existing.insert(pair_key(e[0], e[1]));
        // The two faces of each edge, in the order of the faces.
        std::vector<std::array<int, 2>> edge_faces(mesh.n_edges(), {-1, -1});
        for (int f = 0; f < mesh.n_faces(); f++)
            for (int e : mesh.face_edges[f])
                if (mesh.edge_face_count[e] == 2) (edge_faces[e][0] < 0 ? edge_faces[e][0] : edge_faces[e][1]) = f;

        std::vector<char> touched(mesh.n_faces(), 0);
        int done = 0;
        for (int edge = 0; edge < mesh.n_edges(); edge++) {
            if (mesh.edge_face_count[edge] != 2) continue;
            int g1 = edge_faces[edge][0], g2 = edge_faces[edge][1];
            if (is_input[g1] || is_input[g2]) continue;
            if (touched[g1] || touched[g2]) continue;
            const int u = mesh.edges[edge][0], v = mesh.edges[edge][1];
            auto rotated = [&](int g) {
                Tri t = F[g];
                for (int r = 0; r < 2 && t[0] != u; r++) t = {t[1], t[2], t[0]};
                return t;
            };
            Tri t = rotated(g1);
            if (t[1] != v) {  // g1 must traverse u -> v
                std::swap(g1, g2);
                t = rotated(g1);
            }
            const int a = t[0], b = t[1], c = t[2];
            int d = -1;
            for (int x : F[g2])
                if (x != a && x != b) { d = x; break; }
            if (d < 0) continue;
            if (existing.count(pair_key(c, d))) continue;
            if (angle_at(V[c], V[a], V[b]) + angle_at(V[d], V[a], V[b]) <= M_PI) continue;
            const Vec3 old = cross(V[b] - V[a], V[c] - V[a]) + cross(V[a] - V[b], V[d] - V[b]);
            const Vec3 n1 = cross(V[d] - V[a], V[c] - V[a]);
            const Vec3 n2 = cross(V[b] - V[d], V[c] - V[d]);
            if (dot(n1, old) <= 0 || dot(n2, old) <= 0) continue;
            F[g1] = {a, d, c};
            F[g2] = {d, b, c};
            existing.erase(pair_key(a, b));
            existing.insert(pair_key(c, d));
            touched[g1] = touched[g2] = 1;
            done++;
        }
        if (!done) break;
    }
}

void erase_faces(std::vector<Tri>& F, std::vector<char>& is_input, const std::vector<char>& remove) {
    std::size_t k = 0;
    for (std::size_t f = 0; f < F.size(); f++)
        if (!remove[f]) { F[k] = F[f]; is_input[k] = is_input[f]; k++; }
    F.resize(k);
    is_input.resize(k);
}

// Removes the free vertices of valence 3 and leaves the triangle of their neighbors: when fairing,
// their three faces fold over one another. The removed vertex is marked fixed so that the system
// does not become singular.
void remove_valence_3(std::vector<Tri>& F, std::vector<char>& is_input, std::vector<char>& fixed) {
    const int n = static_cast<int>(fixed.size());
    std::vector<int> valence(n, 0);
    {
        std::vector<std::uint64_t> keys;
        keys.reserve(3 * F.size());
        for (const Tri& t : F)
            for (int i = 0; i < 3; i++) keys.push_back(pair_key(t[i], t[(i + 1) % 3]));
        std::sort(keys.begin(), keys.end());
        keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
        for (std::uint64_t k : keys) {
            valence[static_cast<int>(k >> 32)]++;
            valence[static_cast<int>(k & 0xffffffffu)]++;
        }
    }
    std::vector<int> candidates;
    for (int v = 0; v < n; v++)
        if (valence[v] == 3 && !fixed[v]) candidates.push_back(v);
    if (candidates.empty()) return;

    std::unordered_set<Tri, TriHash> existing;
    existing.reserve(2 * F.size());
    for (const Tri& t : F) existing.insert(sorted(t));
    std::vector<std::vector<int>> incident(n);
    for (int f = 0; f < static_cast<int>(F.size()); f++)
        for (int v : F[f]) incident[v].push_back(f);

    std::vector<char> remove(F.size(), 0);
    for (int v : candidates) {
        const std::vector<int>& faces = incident[v];
        if (faces.size() != 3) continue;
        if (remove[faces[0]] || remove[faces[1]] || remove[faces[2]]) continue;
        if (is_input[faces[0]] || is_input[faces[1]] || is_input[faces[2]]) continue;
        // Each face, starting at v, contributes the edge a -> b of the ring.
        int from[3], to[3], count = 0;
        bool valid = true;
        for (int f : faces) {
            const Tri& t = F[f];
            int k = 0;
            while (k < 3 && t[k] != v) k++;
            if (k == 3) { valid = false; break; }
            const int key = t[(k + 1) % 3], value = t[(k + 2) % 3];
            int slot = 0;
            while (slot < count && from[slot] != key) slot++;
            if (slot == count) from[count++] = key;
            to[slot] = value;
        }
        if (!valid) continue;
        auto successor = [&](int key) {
            for (int s = 0; s < count; s++)
                if (from[s] == key) return to[s];
            return -1;
        };
        const int a = from[0];
        const int b = successor(a);
        const int c = b < 0 ? -1 : successor(b);
        if (c < 0 || successor(c) != a) continue;
        const Tri key = sorted(Tri{a, b, c});
        if (existing.count(key)) continue;
        existing.insert(key);
        F[faces[0]] = {a, b, c};
        remove[faces[1]] = remove[faces[2]] = 1;
        fixed[v] = 1;
    }
    erase_faces(F, is_input, remove);
}

// Collapses the shortest edge of the triangles of the caps with an angle smaller than min_angle
// degrees, only if the endpoints share exactly two neighbors (link condition), so that the surface
// remains a manifold.
void collapse_needles(const Points& V, std::vector<Tri>& F, std::vector<char>& is_input, std::vector<char>& fixed, double min_angle = 10.0) {
    const Mesh mesh(V, F);
    const std::vector<std::array<double, 3>> angles = mesh.corner_angles();
    std::vector<int> needles;
    for (int f = 0; f < mesh.n_faces(); f++) {
        const double smallest = std::min(angles[f][0], std::min(angles[f][1], angles[f][2]));
        if (!is_input[f] && smallest * (180.0 / M_PI) < min_angle) needles.push_back(f);
    }
    if (needles.empty()) return;

    const int n = mesh.n_vertices();
    std::vector<std::vector<int>> neighbors(n), incident(n);
    for (const Edge& e : mesh.edges) {
        neighbors[e[0]].push_back(e[1]);
        neighbors[e[1]].push_back(e[0]);
    }
    for (std::vector<int>& list : neighbors) std::sort(list.begin(), list.end());
    for (int f = 0; f < mesh.n_faces(); f++)
        for (int v : F[f]) incident[v].push_back(f);

    std::vector<char> remove(F.size(), 0), touched(n, 0);
    for (int f : needles) {
        if (remove[f]) continue;
        const Tri t = F[f];
        int k = 0;
        double shortest = INFINITY;
        for (int i = 0; i < 3; i++) {
            const double length = norm(V[t[(i + 2) % 3]] - V[t[(i + 1) % 3]]);
            if (length < shortest) { shortest = length; k = i; }
        }
        int a = t[(k + 1) % 3], b = t[(k + 2) % 3];
        if (fixed[a] && fixed[b]) continue;
        if (fixed[a]) std::swap(a, b);  // a disappears, so it must be free
        if (touched[a] || touched[b]) continue;
        int common = 0;
        for (std::size_t i = 0, j = 0; i < neighbors[a].size() && j < neighbors[b].size();) {
            if (neighbors[a][i] < neighbors[b][j]) i++;
            else if (neighbors[a][i] > neighbors[b][j]) j++;
            else { common++; i++; j++; }
        }
        if (common != 2) continue;
        bool on_input_face = false;
        std::vector<int> faces_a;
        for (int g : incident[a])
            if (!remove[g]) { faces_a.push_back(g); on_input_face |= is_input[g] != 0; }
        if (on_input_face) continue;
        for (int g : faces_a) {
            if (F[g][0] == b || F[g][1] == b || F[g][2] == b) remove[g] = 1;
            else
                for (int& x : F[g])
                    if (x == a) x = b;
        }
        fixed[a] = 1;
        touched[a] = touched[b] = 1;
        for (int x : neighbors[a]) touched[x] = 1;
    }
    erase_faces(F, is_input, remove);
}

}  // namespace

void fair_filled_patches(const Points& V_in, const std::vector<Tri>& F_in, const Mesh& surf, Points& out_V, std::vector<Tri>& out_F,
                         int refine, int rounds, int order, FairingStats* stats) {
    if (stats) *stats = FairingStats();
    Points V = V_in;
    std::vector<Tri> F = F_in;
    Vec3 lo, hi;
    surf.bounds(lo, hi);
    const double tol = 1e-9 * norm(hi - lo);

    std::vector<char> fixed = on_input(V, surf, tol);
    Points centers(F.size());
    for (std::size_t f = 0; f < F.size(); f++) centers[f] = (V[F[f][0]] + V[F[f][1]] + V[F[f][2]]) / 3.0;
    const std::vector<char> center_on = on_input(centers, surf, tol);
    std::vector<char> is_input(F.size());
    bool all_input = true;
    for (std::size_t f = 0; f < F.size(); f++) {
        is_input[f] = fixed[F[f][0]] && fixed[F[f][1]] && fixed[F[f][2]] && center_on[f];
        all_input &= is_input[f] != 0;
    }
    if (all_input) {
        out_V = V;
        out_F = F;
        return;
    }

    for (int level = 0; level <= refine; level++) {
        if (level > 0) split_caps(V, F, is_input, fixed);
        for (int r = 0; r < rounds; r++) {
            V = smooth(V, F, fixed, order);
            delaunay_flips(V, F, is_input);
            remove_valence_3(F, is_input, fixed);
            collapse_needles(V, F, is_input, fixed);
        }
    }
    V = smooth(V, F, fixed, order, stats);
    const Mesh out = compact(V, F);
    out_V = out.V;
    out_F = out.F;
}

}  // namespace gsd
