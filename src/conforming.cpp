#include "gsd/conforming.h"

#include <algorithm>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <string>

#include "gsd/kdtree.h"
#include "tetgen.h"

namespace gsd {
namespace {

// Sign of the volume of abcd, with 0 if it is of the order of rounding. Without the tolerance,
// neighboring faces (which share vertices) would appear to intersect.
double orient3d_tol(const Vec3& a, const Vec3& b, const Vec3& c, const Vec3& d) {
    const Vec3 u = b - a, v = c - a, w = d - a;
    const double vol = dot(u, cross(v, w));
    const double scale = norm(u) * norm(v) * norm(w);
    if (std::fabs(vol) <= 1e-10 * scale) return 0.0;
    return vol > 0 ? 1.0 : -1.0;
}

// Whether the segment PQ crosses the triangle ABC (its boundary included).
bool edge_crosses(const Vec3& P, const Vec3& Q, const Vec3& A, const Vec3& B, const Vec3& C) {
    const double sides = orient3d_tol(A, B, C, P) * orient3d_tol(A, B, C, Q);
    if (!(sides < 0)) return false;
    const double t1 = orient3d_tol(P, Q, A, B), t2 = orient3d_tol(P, Q, B, C), t3 = orient3d_tol(P, Q, C, A);
    return (t1 >= 0 && t2 >= 0 && t3 >= 0) || (t1 <= 0 && t2 <= 0 && t3 <= 0);
}

// What the codes that TetGen throws mean (terminatetetgen in tetgen.h).
const char* tetgen_reason(int code) {
    switch (code) {
        case 1: return "out of memory";
        case 2: return "internal error";
        case 3: return "the input surface has self-intersections";
        case 4: return "a very small input feature was found";
        case 5: return "two input facets are very close";
        case 10: return "the input is not valid";
        default: return "unknown error";
    }
}

std::vector<Tet> run_tetgen(const Points& V, const std::vector<Tri>& F, Points& out_points) {
    // TetGen reads past its arrays with fewer points than a tetrahedron has.
    if (V.size() < 4) throw std::invalid_argument("the tetrahedralization needs at least 4 vertices");
    tetgenio in, out, addin, bgmin;
    in.firstnumber = 0;
    in.numberofpoints = static_cast<int>(V.size());
    in.pointlist = new REAL[3 * V.size()];
    for (std::size_t i = 0; i < V.size(); i++) {
        in.pointlist[3 * i] = V[i].x;
        in.pointlist[3 * i + 1] = V[i].y;
        in.pointlist[3 * i + 2] = V[i].z;
    }
    in.numberoffacets = static_cast<int>(F.size());
    in.facetlist = new tetgenio::facet[F.size()];
    for (std::size_t f = 0; f < F.size(); f++) {
        tetgenio::facet& facet = in.facetlist[f];
        tetgenio::init(&facet);
        facet.numberofpolygons = 1;
        facet.polygonlist = new tetgenio::polygon[1];
        tetgenio::polygon& poly = facet.polygonlist[0];
        tetgenio::init(&poly);
        poly.numberofvertices = 3;
        poly.vertexlist = new int[3];
        for (int j = 0; j < 3; j++) poly.vertexlist[j] = F[f][j];
    }
    // p: PLC, YY: no new points on the input faces, c: keeps the convex hull, Q: quiet.
    tetgenbehavior behavior;
    char switches[] = "pYYcQ";
    if (!behavior.parse_commandline(switches)) throw std::runtime_error("TetGen: unable to parse switches");
    try {
        tetrahedralize(&behavior, &in, &out, &addin, &bgmin);
    } catch (int code) {
        throw std::runtime_error(std::string("TetGen failed: ") + tetgen_reason(code));
    } catch (const std::exception& e) {  // this release of TetGen throws its messages
        throw std::runtime_error(std::string("TetGen failed: ") + e.what());
    } catch (...) {
        throw std::runtime_error("TetGen failed");
    }
    out_points.resize(out.numberofpoints);
    for (int i = 0; i < out.numberofpoints; i++)
        out_points[i] = {out.pointlist[3 * i], out.pointlist[3 * i + 1], out.pointlist[3 * i + 2]};
    std::vector<Tet> tets(out.numberoftetrahedra);
    for (int t = 0; t < out.numberoftetrahedra; t++)
        for (int j = 0; j < 4; j++) tets[t][j] = out.tetrahedronlist[out.numberofcorners * t + j] - out.firstnumber;
    return tets;
}

}  // namespace

std::vector<char> self_intersecting_faces(const Mesh& surf) {
    const int m = surf.n_faces();
    std::vector<char> mask(m, 0);
    if (m < 2) return mask;
    std::vector<Vec3> lo(m), hi(m);
    for (int f = 0; f < m; f++) {
        lo[f] = min(min(surf.V[surf.F[f][0]], surf.V[surf.F[f][1]]), surf.V[surf.F[f][2]]);
        hi[f] = max(max(surf.V[surf.F[f][0]], surf.V[surf.F[f][1]]), surf.V[surf.F[f][2]]);
    }
    // Sweep over the boxes along the axis where the mesh is widest: every pair of faces whose
    // boxes overlap is tested.
    Vec3 mesh_lo, mesh_hi;
    surf.bounds(mesh_lo, mesh_hi);
    const Vec3 size = mesh_hi - mesh_lo;
    const int s0 = size.x >= size.y && size.x >= size.z ? 0 : (size.y >= size.z ? 1 : 2);
    const int s1 = (s0 + 1) % 3, s2 = (s0 + 2) % 3;
    std::vector<int> order(m);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) { return lo[a][s0] < lo[b][s0]; });
#pragma omp parallel for schedule(dynamic, 64)
    for (int a = 0; a < m; a++) {
        const int i = order[a];
        const Vec3 Ti[3] = {surf.V[surf.F[i][0]], surf.V[surf.F[i][1]], surf.V[surf.F[i][2]]};
        for (int b = a + 1; b < m && lo[order[b]][s0] <= hi[i][s0]; b++) {
            const int j = order[b];
            if (!(lo[i][s1] <= hi[j][s1] && lo[j][s1] <= hi[i][s1] && lo[i][s2] <= hi[j][s2] && lo[j][s2] <= hi[i][s2])) continue;
            const Vec3 Tj[3] = {surf.V[surf.F[j][0]], surf.V[surf.F[j][1]], surf.V[surf.F[j][2]]};
            bool crossing = false;
            for (int k = 0; k < 3 && !crossing; k++) {
                const int s = (k + 1) % 3;
                crossing = edge_crosses(Ti[k], Ti[s], Tj[0], Tj[1], Tj[2]) || edge_crosses(Tj[k], Tj[s], Ti[0], Ti[1], Ti[2]);
            }
            if (crossing) {
#pragma omp atomic write
                mask[i] = 1;
#pragma omp atomic write
                mask[j] = 1;
            }
        }
    }
    return mask;
}

Mesh red_green_refine(const Mesh& mesh, std::vector<char> marked) {
    std::vector<char> split(mesh.n_edges(), 0);
    while (true) {
        std::fill(split.begin(), split.end(), 0);
        for (int f = 0; f < mesh.n_faces(); f++)
            if (marked[f])
                for (int e : mesh.face_edges[f]) split[e] = 1;
        bool any = false;
        for (int f = 0; f < mesh.n_faces(); f++) {
            if (marked[f]) continue;
            const int n = split[mesh.face_edges[f][0]] + split[mesh.face_edges[f][1]] + split[mesh.face_edges[f][2]];
            if (n >= 2) { marked[f] = 1; any = true; }
        }
        if (!any) break;
    }
    return mesh.split_edges(split);
}

TetMesh conforming_tet_mesh(const Mesh& surf, std::vector<int>* ids, int refine_intersections) {
    // The faces that do not intersect go to the constraint. Subdividing does not change the
    // indices of the input vertices (the new ones go at the end).
    struct Attempt { Points V; std::vector<Tri> F; };
    std::vector<Attempt> attempts;
    auto constrained = [](const Mesh& m, const std::vector<char>& crossing) {
        std::vector<Tri> F;
        for (int f = 0; f < m.n_faces(); f++)
            if (!crossing[f]) F.push_back(m.F[f]);
        return F;
    };
    Mesh current = surf;
    std::vector<char> crossing = self_intersecting_faces(current);
    attempts.push_back({current.V, constrained(current, crossing)});
    for (int r = 0; r < refine_intersections; r++) {
        if (std::none_of(crossing.begin(), crossing.end(), [](char c) { return c != 0; })) break;
        current = red_green_refine(current, crossing);
        crossing = self_intersecting_faces(current);
        attempts.push_back({current.V, constrained(current, crossing)});
    }

    Points points;
    std::vector<Tet> tets;
    for (int k = static_cast<int>(attempts.size()) - 1; k >= 0; k--) {
        try {
            tets = run_tetgen(attempts[k].V, attempts[k].F, points);
            break;
        } catch (const std::runtime_error&) {
            if (k == 0) throw;
        }
    }
    TetMesh mesh(points, drop_degenerate_tets(points, tets));
    if (ids) *ids = KdTree(mesh.V).nearest_indices(surf.V);
    return mesh;
}

}  // namespace gsd
