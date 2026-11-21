// Filling the holes by mirroring the input across its symmetry plane.
//
// Fairing cannot invent detail: if a hole takes away the nose, the cap stays smooth. If the model
// is symmetric, what is missing on one side is usually on the other. The plane that minimizes the
// distance from the mirrored surface to the original is sought, ignoring the 20% of points that
// match worst (the holes). The mirrored faces that fall where the input has no surface and that
// are connected with some face close to the fill of a first pass are taken; the second condition
// avoids adding pieces that are not symmetric in the pose. The reconstruction is then run again
// with those faces added to the input, without stitching them. If the residual of the plane
// exceeds half an edge, the model is taken as not symmetric and nothing is done.
#pragma once
#include <cstdint>
#include <vector>

#include "gsd/fairing.h"
#include "gsd/mesh.h"
#include "gsd/rng.h"

namespace gsd {

// k points uniformly distributed on the surface, by area.
Points sample_surface(const Points& V, const std::vector<Tri>& F, int k, Rng& rng);

struct Plane {
    Vec3 n;
    double d = 0.0;         // n . x = d
    double residual = 0.0;  // mean distance from the mirrored surface to the original over the best fraction
};

// Symmetry plane of surf. The search starts from each principal axis of the samples; axes
// replaces them when given (three unit vectors).
Plane symmetry_plane(const Mesh& surf, int n_samples = 4000, double trim = 0.8, std::uint64_t seed = 0, const std::vector<Vec3>* axes = nullptr);

// Faces of surf mirrored that fall into its holes. filled is a first reconstruction. The mirrored
// faces farther than far edges from the input and connected (through faces of that kind) with
// some face closer than near times the side of the box of the fill are taken. Mirroring reverses
// the orientation, so they are flipped.
Mesh mirror_completion(const Mesh& surf, const Mesh& filled, const Vec3& n, double d, double far = 2.0, double near = 0.06, std::uint64_t seed = 0);

struct Reconstruction {
    Points method_V, faired_V, symmetry_V;
    std::vector<Tri> method_F, faired_F, symmetry_F;
    bool has_plane = false, has_symmetry = false;
    Plane plane;
    Mesh mirrored;
    FairingStats fairing, fairing_second;  // how the fairing of each pass ended
};

// Segmentation, fairing and, if the model is symmetric, a second pass with the input completed by
// reflection. axes replaces the starting axes of the search for the plane, as in symmetry_plane.
// The tolerances and the weight of the smoothness term are absolute, tuned for models about one
// unit across, so the result depends on the size of the model.
Reconstruction reconstruct_with_symmetry(const Mesh& surf, bool symmetry = true, double max_residual = 0.5, double gamma = 40.0, int quadrature = 1,
                                         const std::vector<Vec3>* axes = nullptr);

// Median of an array, as numpy.median.
double median(std::vector<double> values);

}  // namespace gsd
