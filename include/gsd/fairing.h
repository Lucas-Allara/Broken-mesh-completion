// Fairing of the caps added by the segmentation.
//
// Where there was a hole, the output surface goes through faces of the tetrahedralization and ends
// up wrinkled. Here the points that lie on the input are kept fixed and those of the caps are
// relocated so that they cancel the tri-Laplacian, which blends with what surrounds the hole in
// position, tangent and curvature. Before that, the connectivity of the caps is fixed with
// Delaunay flips and a subdivision. The input faces are not touched, nor are the tetrahedral mesh
// and the labels: it is a post-process of the surface.
#pragma once
#include <vector>

#include "gsd/mesh.h"

namespace gsd {

// How the last solve of the fairing ended: the number of caps, and how many of them were solved
// with each power of the Laplacian (by_order[k] for the power k). A cap that leaves the box of its
// anchors is solved again with one power less.
struct FairingStats {
    int caps = 0;
    int by_order[4] = {0, 0, 0, 0};
};

// Fairs the caps of (V, F) keeping fixed what lies on surf. refine is how many times the caps are
// subdivided, rounds how many fairing and flip passes are done per level and order the power of
// the Laplacian (2 is the thin plate). stats, when given, receives how the last solve ended.
void fair_filled_patches(const Points& V, const std::vector<Tri>& F, const Mesh& surf, Points& out_V, std::vector<Tri>& out_F,
                         int refine = 2, int rounds = 3, int order = 3, FairingStats* stats = nullptr);

// Which points of P lie on some triangle of surf, within tol.
std::vector<char> on_input(const Points& P, const Mesh& surf, double tol);

}  // namespace gsd
