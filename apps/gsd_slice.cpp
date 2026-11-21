// The generalized winding number on a planar slice, as the figure of the introduction of the
// poster draws it.
//
//     gsd_slice (INPUT.obj | --sphere HOLE) OUTPUT.txt [--axis x|y|z] [--resolution N] [--padding P]
//
// --sphere takes, in place of a file, the unit sphere without the faces whose center lies closer
// than HOLE to its north pole (0 leaves it closed). The slice goes through the origin, across the
// given axis (y by default), and covers the bounding box of the mesh enlarged by P times its
// longest side (0.4 by default) with N by N points (320 by default). The first line of OUTPUT.txt
// holds N and the limits of the slice along its two axes, and each of the following N lines one
// row of values of w, from the lowest second coordinate to the highest.
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

#include "gsd/mesh.h"
#include "gsd/shapes.h"
#include "gsd/winding.h"

int main(int argc, char** argv) {
    std::vector<std::string> files;
    bool sphere = false, valid = true;
    double hole = 0.0, padding = 0.4;
    int axis = 1, resolution = 320;
    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        const bool has_value = i + 1 < argc;
        if (a == "--sphere" && has_value) {
            sphere = true;
            hole = std::atof(argv[++i]);
        } else if (a == "--axis" && has_value) {
            const std::string name = argv[++i];
            axis = name == "x" ? 0 : name == "y" ? 1 : name == "z" ? 2 : -1;
        } else if (a == "--resolution" && has_value) resolution = std::atoi(argv[++i]);
        else if (a == "--padding" && has_value) padding = std::atof(argv[++i]);
        else if (a.rfind("--", 0) == 0) valid = false;
        else files.push_back(a);
    }
    if (!valid || axis < 0 || resolution < 2 || padding < 0 || hole < 0 || files.size() != (sphere ? 1u : 2u)) {
        std::fprintf(stderr, "usage: gsd_slice (INPUT.obj | --sphere HOLE) OUTPUT.txt [--axis x|y|z] [--resolution N] [--padding P]\n");
        return 1;
    }
    try {
        gsd::Mesh mesh = sphere ? gsd::icosphere(4) : gsd::load_obj(files[0]);
        if (sphere && hole > 0) mesh = gsd::punch_hole(mesh, {0.0, 0.0, 1.0}, hole);
        if (mesh.n_faces() == 0) throw std::runtime_error("the mesh has no faces");
        gsd::Vec3 lo, hi;
        mesh.bounds(lo, hi);
        const double margin = padding * std::max(hi.x - lo.x, std::max(hi.y - lo.y, hi.z - lo.z));
        const int u = axis == 0 ? 1 : 0, v = axis == 2 ? 1 : 2;  // the other two axes, in their order
        const double u0 = lo[u] - margin, u1 = hi[u] + margin, v0 = lo[v] - margin, v1 = hi[v] + margin;
        const std::size_t n = static_cast<std::size_t>(resolution);
        gsd::Points points(n * n);
        for (std::size_t j = 0; j < n; j++)
            for (std::size_t i = 0; i < n; i++) {
                gsd::Vec3 p;
                p[u] = u0 + (u1 - u0) * static_cast<double>(i) / static_cast<double>(n - 1);
                p[v] = v0 + (v1 - v0) * static_cast<double>(j) / static_cast<double>(n - 1);
                points[j * n + i] = p;
            }
        const std::vector<double> w = gsd::winding_number(mesh, points);

        const std::string& output = files.back();
        std::FILE* out = std::fopen(output.c_str(), "w");
        if (!out) throw std::runtime_error("cannot write " + output);
        std::fprintf(out, "%d %.17g %.17g %.17g %.17g\n", resolution, u0, u1, v0, v1);
        for (std::size_t j = 0; j < n; j++) {
            for (std::size_t i = 0; i < n; i++) std::fprintf(out, i ? " %.9g" : "%.9g", w[j * n + i]);
            std::fprintf(out, "\n");
        }
        std::fclose(out);
        const auto range = std::minmax_element(w.begin(), w.end());
        std::printf("%d faces, w from %.3f to %.3f\n", mesh.n_faces(), *range.first, *range.second);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "gsd_slice: %s\n", e.what());
        return 1;
    }
    return 0;
}
