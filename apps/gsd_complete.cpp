// Completes a broken mesh: the method of the paper, the fairing of the caps and, if the model is
// symmetric, the second pass with the mirrored faces.
//
//     gsd_complete input.obj output.obj [--method | --fairing] [--all PREFIX]
//
// output.obj is the result with symmetry when a symmetry plane is found and mirrored faces fall in
// the holes, and the faired one otherwise; --method and --fairing stop at those stages, and --all
// also writes PREFIX_method.obj, PREFIX_faired.obj and, when the second pass ran,
// PREFIX_symmetry.obj.
//
// The tolerances of the pipeline are absolute and tuned for models about one unit across, so the
// model is centered and scaled to the size that the experiment uses, and the results are taken
// back to the coordinates of the input, whose vertices come out bit for bit.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "gsd/mesh.h"
#include "gsd/symmetry.h"

namespace {

const double kSize = 1.35;  // longest side of the models in the experiment

struct PointHash {
    std::size_t operator()(const gsd::Vec3& p) const {
        std::uint64_t h = 1469598103934665603ull, bits;
        for (int a = 0; a < 3; a++) {
            const double v = p[a];
            std::memcpy(&bits, &v, sizeof bits);
            h = (h ^ bits) * 1099511628211ull;
        }
        return static_cast<std::size_t>(h);
    }
};
struct PointEqual {
    bool operator()(const gsd::Vec3& a, const gsd::Vec3& b) const { return a.x == b.x && a.y == b.y && a.z == b.z; }
};

}  // namespace

int main(int argc, char** argv) {
    std::string input, output, prefix;
    int stage = 2;  // 0 method, 1 fairing, 2 symmetry
    bool valid = true;
    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        if (a == "--method") stage = 0;
        else if (a == "--fairing") stage = 1;
        else if (a == "--all" && i + 1 < argc) prefix = argv[++i];
        else if (a.rfind("--", 0) == 0) valid = false;
        else if (input.empty()) input = a;
        else if (output.empty()) output = a;
        else valid = false;
    }
    if (!valid || input.empty() || output.empty()) {
        std::fprintf(stderr, "usage: gsd_complete input.obj output.obj [--method | --fairing] [--all PREFIX]\n");
        return 1;
    }
    try {
        const gsd::Mesh surf = gsd::load_obj(input);
        gsd::Vec3 lo, hi;
        surf.bounds(lo, hi);
        const gsd::Vec3 center = (lo + hi) / 2.0;
        const double longest = std::max(hi.x - lo.x, std::max(hi.y - lo.y, hi.z - lo.z));
        if (!(longest > 0)) throw std::runtime_error(input + ": the model has no extent");
        const double scale = kSize / longest;
        gsd::Points V(surf.V.size());
        std::unordered_map<gsd::Vec3, gsd::Vec3, PointHash, PointEqual> original;
        original.reserve(2 * V.size());
        for (std::size_t i = 0; i < V.size(); i++) {
            V[i] = (surf.V[i] - center) * scale;
            original.emplace(V[i], surf.V[i]);
        }
        // Back to the coordinates of the input: its own vertices exactly, the new ones by the inverse.
        auto restore = [&](gsd::Points P) {
            for (gsd::Vec3& p : P) {
                const auto it = original.find(p);
                p = it != original.end() ? it->second : p / scale + center;
            }
            return P;
        };

        const auto t0 = std::chrono::steady_clock::now();
        const gsd::Reconstruction r = gsd::reconstruct_with_symmetry(gsd::Mesh(V, surf.F), stage == 2);
        const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        if (stage == 0) gsd::save_obj(restore(r.method_V), r.method_F, output);
        else if (stage == 1 || !r.has_symmetry) gsd::save_obj(restore(r.faired_V), r.faired_F, output);
        else gsd::save_obj(restore(r.symmetry_V), r.symmetry_F, output);
        if (!prefix.empty()) {
            gsd::save_obj(restore(r.method_V), r.method_F, prefix + "_method.obj");
            gsd::save_obj(restore(r.faired_V), r.faired_F, prefix + "_faired.obj");
            if (r.has_symmetry) gsd::save_obj(restore(r.symmetry_V), r.symmetry_F, prefix + "_symmetry.obj");
        }
        std::printf("input %d faces, method %zu, faired %zu", surf.n_faces(), r.method_F.size(), r.faired_F.size());
        if (r.has_symmetry) std::printf(", with symmetry %zu (%d mirrored faces)", r.symmetry_F.size(), r.mirrored.n_faces());
        else if (r.has_plane) std::printf(", symmetry not used");
        std::printf(", %.2f s\n", seconds);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "gsd_complete: %s\n", e.what());
        return 1;
    }
    return 0;
}
