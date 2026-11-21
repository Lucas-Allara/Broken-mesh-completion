// The experiment of the results: the error in the holes for the method of the paper, with fairing
// and with fairing and symmetry, on Spot and Homer with 5%, 10% and 22% of the surface removed
// (three sets of five holes per level), and fairing with the Laplacian, with the bi-Laplacian and
// with three subdivisions on the same output of the method.
//
//     gsd_experiment REPOSITORY OUTPUT.json [--meshes DIRECTORY] [--axes FILE]
//
// Each row of OUTPUT.json is one case: the removed fraction, the seconds of the reconstruction,
// the errors and the floor of the measure (the original measured against itself), the symmetry
// plane and the number of mirrored faces, how many caps the fairing of each pass solved with each
// power of the Laplacian, and the boundary edges, the non-manifold edges and the connected pieces
// of every output. --meshes also saves the meshes of every case, the mirrored faces and the two
// models as the experiment scales them, which poster/poster_figures.py renders. --axes gives the
// starting axes of the search for the symmetry plane, one case per line (its name and the nine
// numbers of the three axes); tests/reference_axes.txt holds those of the reference values of the
// tests, which are the same principal axes with other signs.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#include "gsd/fairing.h"
#include "gsd/kdtree.h"
#include "gsd/mesh.h"
#include "gsd/rng.h"
#include "gsd/shapes.h"
#include "gsd/symmetry.h"

using namespace gsd;

namespace {

const double kSize = 1.35;

Mesh load_model(const std::string& path, bool permute) {
    const Mesh m = load_obj(path);
    Vec3 lo, hi;
    m.bounds(lo, hi);
    const Vec3 center = (lo + hi) / 2.0;
    const double scale = kSize / std::max(hi.x - lo.x, std::max(hi.y - lo.y, hi.z - lo.z));
    Points V(m.V.size());
    for (std::size_t i = 0; i < V.size(); i++) {
        const Vec3 p = (m.V[i] - center) * scale;
        V[i] = permute ? Vec3{p.x, -p.z, p.y} : p;  // Spot has its vertical axis along y
    }
    return Mesh(std::move(V), m.F);
}

// Mean distance from the missing true surface to the reconstruction, as a percentage of the size
// of the model.
double hole_error(const Points& missing_truth, const Points& V, const std::vector<Tri>& F, Rng& rng) {
    const KdTree tree(sample_surface(V, F, 300000, rng));
    const std::vector<double> d = tree.nearest_distances(missing_truth);
    return 100.0 * (pairwise_sum_of(d) / static_cast<double>(d.size())) / kSize;
}

// The size and the defects of a mesh, as a JSON object.
std::string describe(const Points& V, const std::vector<Tri>& F) {
    const Mesh m = compact(V, F);
    int boundary = 0, nonmanifold = 0;
    for (int count : m.edge_face_count) {
        boundary += count == 1;
        nonmanifold += count > 2;
    }
    char text[160];
    std::snprintf(text, sizeof text, "{\"faces\": %d, \"boundary_edges\": %d, \"nonmanifold_edges\": %d, \"pieces\": %d}", m.n_faces(),
                  boundary, nonmanifold, connected_components(m));
    return text;
}

// The caps of a fairing and how many of them ended with the powers 1, 2 and 3 of the Laplacian.
std::string describe(const FairingStats& s) {
    char text[80];
    std::snprintf(text, sizeof text, "{\"caps\": %d, \"by_order\": [%d, %d, %d]}", s.caps, s.by_order[1], s.by_order[2], s.by_order[3]);
    return text;
}

std::map<std::string, std::vector<Vec3>> read_axes(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open " + path);
    std::map<std::string, std::vector<Vec3>> axes;
    std::string line;
    while (std::getline(in, line)) {
        std::istringstream ss(line);
        std::string key;
        std::vector<Vec3> a(3);
        if (!(ss >> key)) continue;
        for (Vec3& v : a)
            if (!(ss >> v.x >> v.y >> v.z)) throw std::runtime_error(path + ": malformed line for " + key);
        axes[key] = a;
    }
    return axes;
}

int run(int argc, char** argv) {
    const std::string repo = argv[1], output = argv[2];
    std::string mesh_dir;
    std::map<std::string, std::vector<Vec3>> axes;
    bool with_axes = false;
    for (int i = 3; i + 1 < argc; i += 2) {
        const std::string option = argv[i];
        if (option == "--meshes") mesh_dir = argv[i + 1];
        else if (option == "--axes") { axes = read_axes(argv[i + 1]); with_axes = true; }
        else throw std::runtime_error("unknown option " + option);
    }
    if (!mesh_dir.empty()) std::filesystem::create_directories(mesh_dir);

    struct Model { const char* name; Mesh truth; };
    const Model models[] = {{"spot", load_model(repo + "/assets/spot/spot_triangulated.obj", true)},
                            {"homer", load_model(repo + "/assets/homer.obj", false)}};
    const double levels[] = {0.05, 0.10, 0.22};
    const int seeds[] = {3, 7, 21};

    std::FILE* out = std::fopen(output.c_str(), "w");
    if (!out) throw std::runtime_error("cannot write " + output);
    std::fprintf(out, "[\n");
    bool first = true;
    for (const Model& model : models) {
        const Mesh& truth = model.truth;
        if (!mesh_dir.empty()) save_obj(truth, mesh_dir + "/" + model.name + "_original.obj");
        for (double fraction : levels)
            for (int seed : seeds) {
                const std::string key = std::string(model.name) + "_" + std::to_string(static_cast<int>(fraction * 100)) + "_" + std::to_string(seed);
                Rng rng(0);
                // Input with holes, without loose vertices, and the missing faces.
                const Mesh holed = random_holes(truth, fraction, 5, seed);
                const Mesh broken = compact(holed.V, holed.F);
                std::unordered_set<Tri, TriHash> present(holed.F.begin(), holed.F.end());
                std::vector<Tri> missing;
                for (const Tri& t : truth.F)
                    if (!present.count(t)) missing.push_back(t);
                const Points missing_points = sample_surface(truth.V, missing, 100000, rng);

                const std::vector<Vec3>* start = nullptr;
                if (with_axes) {
                    const auto it = axes.find(key);
                    if (it == axes.end()) throw std::runtime_error("no axes for " + key);
                    start = &it->second;
                }
                const auto t0 = std::chrono::steady_clock::now();
                const Reconstruction r = reconstruct_with_symmetry(broken, true, 0.5, 40.0, 1, start);
                const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

                // Without mirrored faces there is no second pass, and the result with symmetry is the faired one.
                const Points& symmetry_V = r.has_symmetry ? r.symmetry_V : r.faired_V;
                const std::vector<Tri>& symmetry_F = r.has_symmetry ? r.symmetry_F : r.faired_F;
                const double e_method = hole_error(missing_points, r.method_V, r.method_F, rng);
                const double e_faired = hole_error(missing_points, r.faired_V, r.faired_F, rng);
                const double e_symmetry = hole_error(missing_points, symmetry_V, symmetry_F, rng);
                double alternatives[3];
                const int options[3][2] = {{2, 1}, {2, 2}, {3, 3}};  // (refine, order): Laplacian, bi-Laplacian, three subdivisions
                for (int k = 0; k < 3; k++) {
                    Points V;
                    std::vector<Tri> F;
                    fair_filled_patches(r.method_V, r.method_F, broken, V, F, options[k][0], 3, options[k][1]);
                    alternatives[k] = hole_error(missing_points, V, F, rng);
                }
                // The floor comes last, so that the samples of the errors above are those of the reference values.
                const double floor = hole_error(missing_points, truth.V, truth.F, rng);
                const double removed = 1.0 - broken.total_area() / truth.total_area();
                std::fprintf(out, "%s {\"model\": \"%s\", \"level\": %.17g, \"seed\": %d, \"missing\": %.17g, \"seconds\": %.17g,\n", first ? "" : ",\n",
                             model.name, fraction, seed, removed, seconds);
                std::fprintf(out, "  \"method\": %.17g, \"faired\": %.17g, \"symmetry\": %.17g, \"floor\": %.17g,\n", e_method, e_faired, e_symmetry, floor);
                std::fprintf(out, "  \"laplacian\": %.17g, \"bilaplacian\": %.17g, \"three_subdivisions\": %.17g,\n", alternatives[0], alternatives[1],
                             alternatives[2]);
                std::fprintf(out, "  \"plane\": [%.17g, %.17g, %.17g, %.17g], \"plane_residual\": %.17g, \"mirrored\": %d,\n", r.plane.n.x, r.plane.n.y,
                             r.plane.n.z, r.plane.d, r.plane.residual, r.has_symmetry ? r.mirrored.n_faces() : 0);
                std::fprintf(out, "  \"fairing\": %s, \"fairing_second_pass\": %s,\n", describe(r.fairing).c_str(), describe(r.fairing_second).c_str());
                std::fprintf(out, "  \"input_mesh\": %s,\n  \"method_mesh\": %s,\n  \"faired_mesh\": %s,\n  \"symmetry_mesh\": %s}",
                             describe(broken.V, broken.F).c_str(), describe(r.method_V, r.method_F).c_str(),
                             describe(r.faired_V, r.faired_F).c_str(), describe(symmetry_V, symmetry_F).c_str());
                first = false;
                std::fflush(out);
                std::printf("%s %.0f%% seed %2d: method %.3f%%  faired %.3f%%  symmetry %.3f%%  (%.1fs)\n", model.name, 100 * fraction, seed,
                            e_method, e_faired, e_symmetry, seconds);
                std::fflush(stdout);
                if (!mesh_dir.empty()) {
                    const std::string base = mesh_dir + "/" + key;
                    save_obj(broken, base + "_input.obj");
                    save_obj(r.method_V, r.method_F, base + "_method.obj");
                    save_obj(r.faired_V, r.faired_F, base + "_faired.obj");
                    save_obj(symmetry_V, symmetry_F, base + "_symmetry.obj");
                    if (r.has_symmetry) save_obj(r.mirrored, base + "_mirrored.obj");
                }
            }
    }
    std::fprintf(out, "\n]\n");
    std::fclose(out);
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3 || (argc - 3) % 2 != 0) {
        std::fprintf(stderr, "usage: gsd_experiment REPOSITORY OUTPUT.json [--meshes DIRECTORY] [--axes FILE]\n");
        return 1;
    }
    try {
        return run(argc, argv);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "gsd_experiment: %s\n", e.what());
        return 1;
    }
}
