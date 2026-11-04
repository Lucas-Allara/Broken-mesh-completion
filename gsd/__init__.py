"""Implementation of *Robust Inside-Outside Segmentation using Generalized
Winding Numbers* (Jacobson, Kavan and Sorkine-Hornung, 2013), with fairing of the
caps and symmetry completion."""

from .conforming import conforming_tet_mesh, delaunay_tet_mesh, grid_tet_mesh, self_intersecting_faces
from .fairing import fair_filled_patches
from .mesh import Mesh, load_obj, save_obj
from .segmentation import (
    apply_hard_constraints,
    element_winding_number,
    extract_label_boundary,
    graphcut_segment,
    hard_constraints_from_surface,
    tet_adjacency,
)
from .shapes import (
    capsule,
    connected_components,
    cube,
    ellipsoid,
    humanoid,
    icosphere,
    merge_meshes,
    mobius_strip,
    punch_hole,
    random_holes,
    tetrahedron,
    torus,
)
from .symmetry import mirror_completion, reconstruct_with_symmetry, symmetry_plane
from .tetmesh import TetMesh, drop_degenerate_tets, tet_grid
from .winding import WindingHierarchy, sample_grid, solid_angles, winding_number

__all__ = [
    "Mesh", "load_obj", "save_obj",
    "capsule", "connected_components", "cube", "ellipsoid", "humanoid", "icosphere",
    "merge_meshes", "mobius_strip", "punch_hole", "random_holes", "tetrahedron", "torus",
    "WindingHierarchy", "sample_grid", "solid_angles", "winding_number",
    "TetMesh", "drop_degenerate_tets", "tet_grid",
    "conforming_tet_mesh", "delaunay_tet_mesh", "grid_tet_mesh", "self_intersecting_faces",
    "apply_hard_constraints", "element_winding_number", "extract_label_boundary",
    "graphcut_segment", "hard_constraints_from_surface", "tet_adjacency",
    "fair_filled_patches",
    "mirror_completion", "reconstruct_with_symmetry", "symmetry_plane",
]
