#include "gsd/shapes.h"

#include <cmath>
#include <map>

#include "gsd/rng.h"

namespace gsd {

Mesh icosphere(int subdivisions, double radius) {
    const double phi = (1.0 + std::sqrt(5.0)) / 2.0;
    // Icosahedron with outward-oriented faces.
    Points V = {{-1, phi, 0}, {1, phi, 0}, {-1, -phi, 0}, {1, -phi, 0}, {0, -1, phi}, {0, 1, phi},
                {0, -1, -phi}, {0, 1, -phi}, {phi, 0, -1}, {phi, 0, 1}, {-phi, 0, -1}, {-phi, 0, 1}};
    std::vector<Tri> F = {{0, 11, 5}, {0, 5, 1}, {0, 1, 7}, {0, 7, 10}, {0, 10, 11}, {1, 5, 9}, {5, 11, 4},
                          {11, 10, 2}, {10, 7, 6}, {7, 1, 8}, {3, 9, 4}, {3, 4, 2}, {3, 2, 6}, {3, 6, 8},
                          {3, 8, 9}, {4, 9, 5}, {2, 4, 11}, {6, 2, 10}, {8, 6, 7}, {9, 8, 1}};
    const double scale = norm(V[0]);
    for (Vec3& v : V) v = v / scale;

    for (int s = 0; s < subdivisions; s++) {
        std::map<std::uint64_t, int> midpoint;
        auto mid = [&](int a, int b) {
            const std::uint64_t key = pair_key(a, b);
            auto it = midpoint.find(key);
            if (it != midpoint.end()) return it->second;
            const Vec3 m = (V[a] + V[b]) / 2.0;
            V.push_back(m / norm(m));
            return midpoint[key] = static_cast<int>(V.size()) - 1;
        };
        std::vector<Tri> next;
        next.reserve(4 * F.size());
        for (const Tri& t : F) {
            const int a = t[0], b = t[1], c = t[2];
            const int ab = mid(a, b), bc = mid(b, c), ca = mid(c, a);
            next.push_back({a, ab, ca});
            next.push_back({b, bc, ab});
            next.push_back({c, ca, bc});
            next.push_back({ab, bc, ca});
        }
        F = std::move(next);
    }
    for (Vec3& v : V) v = v * radius;
    return Mesh(std::move(V), std::move(F));
}

Mesh cube(double size) {
    const double h = size / 2.0;
    Points V = {{-h, -h, -h}, {h, -h, -h}, {h, h, -h}, {-h, h, -h}, {-h, -h, h}, {h, -h, h}, {h, h, h}, {-h, h, h}};
    std::vector<Tri> F = {{0, 3, 2}, {0, 2, 1}, {4, 5, 6}, {4, 6, 7}, {0, 1, 5}, {0, 5, 4},
                          {2, 3, 7}, {2, 7, 6}, {0, 4, 7}, {0, 7, 3}, {1, 2, 6}, {1, 6, 5}};
    return Mesh(std::move(V), std::move(F));
}

Mesh box(const Vec3& lo, const Vec3& hi) {
    Mesh c = cube(1.0);
    Points V(c.V.size());
    for (std::size_t i = 0; i < V.size(); i++)
        for (int a = 0; a < 3; a++) V[i][a] = (c.V[i][a] + 0.5) * (hi[a] - lo[a]) + lo[a];
    return Mesh(std::move(V), c.F);
}

Mesh merge_meshes(const std::vector<Mesh>& meshes) {
    Points V;
    std::vector<Tri> F;
    for (const Mesh& m : meshes) {
        const int offset = static_cast<int>(V.size());
        V.insert(V.end(), m.V.begin(), m.V.end());
        for (const Tri& t : m.F) F.push_back({t[0] + offset, t[1] + offset, t[2] + offset});
    }
    return Mesh(std::move(V), std::move(F));
}

Mesh punch_hole(const Mesh& mesh, const Vec3& center, double radius) {
    std::vector<int> drop;
    for (int f = 0; f < mesh.n_faces(); f++)
        if (norm(mesh.face_center(f) - center) < radius) drop.push_back(f);
    return mesh.without_faces(drop);
}

Mesh random_holes(const Mesh& mesh, double target_fraction, int n_holes, std::uint64_t seed) {
    if (target_fraction <= 0) return mesh;
    Rng rng(seed);
    const double initial_area = mesh.total_area();
    Mesh current = mesh;
    const double base_radius = std::sqrt(target_fraction * initial_area / (n_holes * M_PI));
    for (int k = 0; k < n_holes; k++) {
        if (current.n_faces() < 8) break;
        const Vec3 hole_center = current.face_center(static_cast<int>(rng.integers(current.n_faces())));
        const double target = initial_area * target_fraction * (k + 1) / n_holes;
        double hole_radius = base_radius;
        Mesh attempt = current;
        for (int it = 0; it < 24; it++) {
            attempt = punch_hole(current, hole_center, hole_radius);
            if (initial_area - attempt.total_area() >= target || attempt.n_faces() < 8) break;
            hole_radius *= 1.15;
        }
        if (attempt.n_faces() >= 8) current = std::move(attempt);
    }
    return current;
}

}  // namespace gsd
