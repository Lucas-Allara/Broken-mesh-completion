cmake_minimum_required(VERSION 3.16)
project(gsd LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(OpenMP REQUIRED)
# Any Eigen from 3.3 on: the version is checked by hand, since the version file of Eigen 5 rejects a
# request for 3.3.
find_package(Eigen3 REQUIRED NO_MODULE)
if(Eigen3_VERSION VERSION_LESS 3.3)
  message(FATAL_ERROR "Eigen 3.3 or later is needed, found ${Eigen3_VERSION}")
endif()

# TetGen 1.6.0 (AGPL-3.0) is not part of this repository: its three source files are downloaded
# here, from the release 0.8.4 of the tetgen package of PyVista, the build that gave the
# tetrahedralizations of the reference values of the tests.
set(TETGEN_DIR ${CMAKE_CURRENT_BINARY_DIR}/tetgen)
set(TETGEN_URL https://raw.githubusercontent.com/pyvista/tetgen/v0.8.4/src)
file(DOWNLOAD ${TETGEN_URL}/tetgen.h ${TETGEN_DIR}/tetgen.h
     EXPECTED_HASH SHA256=a6dc991343c69613708f5b7a0f7f65e365a2a0f58c91c0d5ebbd4b5c758d7b41)
file(DOWNLOAD ${TETGEN_URL}/tetgen.cxx ${TETGEN_DIR}/tetgen.cxx
     EXPECTED_HASH SHA256=cfc869c80720902ab26052781983d56179717d81c86270a43e3eac92afb20fd4)
file(DOWNLOAD ${TETGEN_URL}/predicates.cxx ${TETGEN_DIR}/predicates.cxx
     EXPECTED_HASH SHA256=10d9ded5821f4b5e3ac91cad9b5ac3a573800b012e9ec58205d12285cb51d75f)
add_library(tetgen STATIC ${TETGEN_DIR}/tetgen.cxx ${TETGEN_DIR}/predicates.cxx)
target_include_directories(tetgen SYSTEM PUBLIC ${TETGEN_DIR})
target_compile_definitions(tetgen PUBLIC TETLIBRARY)
# The flags of that package; the exact predicates must not be contracted or reordered.
target_compile_options(tetgen PRIVATE -O3 -fno-expensive-optimizations -ffp-contract=off -w)

add_library(gsd STATIC
  src/rng.cpp src/kdtree.cpp src/mesh.cpp src/tetmesh.cpp src/winding.cpp src/shapes.cpp
  src/conforming.cpp src/maxflow.cpp src/segmentation.cpp src/constraints.cpp
  src/fairing.cpp src/neldermead.cpp src/symmetry.cpp)
target_include_directories(gsd PUBLIC include)
# Without contraction the sums and products round as in numpy, which keeps the results within
# rounding of the reference values of the tests.
target_compile_options(gsd PRIVATE -Wall -Wextra -ffp-contract=off)
target_link_libraries(gsd PUBLIC OpenMP::OpenMP_CXX PRIVATE tetgen Eigen3::Eigen)

add_executable(gsd_complete apps/gsd_complete.cpp)
target_link_libraries(gsd_complete PRIVATE gsd)
add_executable(gsd_experiment apps/gsd_experiment.cpp)
target_link_libraries(gsd_experiment PRIVATE gsd)
add_executable(gsd_slice apps/gsd_slice.cpp)
target_link_libraries(gsd_slice PRIVATE gsd)

enable_testing()
set(REPOSITORY ${CMAKE_CURRENT_SOURCE_DIR})
foreach(name winding segmentation fairing symmetry)
  add_executable(test_${name} tests/test_${name}.cpp)
  target_link_libraries(test_${name} PRIVATE gsd)
  add_test(NAME ${name} COMMAND test_${name} ${REPOSITORY})
endforeach()
# The program on a whole model, and on two inputs that must end with a message and not with a crash.
add_test(NAME complete COMMAND gsd_complete ${REPOSITORY}/assets/spot/spot_triangulated.obj ${CMAKE_CURRENT_BINARY_DIR}/spot_completed.obj)
add_test(NAME complete_triangle COMMAND gsd_complete ${CMAKE_CURRENT_SOURCE_DIR}/tests/triangle.obj ${CMAKE_CURRENT_BINARY_DIR}/triangle_completed.obj)
set_tests_properties(complete_triangle PROPERTIES PASS_REGULAR_EXPRESSION "at least 4 vertices")
add_test(NAME complete_malformed COMMAND gsd_complete ${CMAKE_CURRENT_SOURCE_DIR}/tests/malformed.obj ${CMAKE_CURRENT_BINARY_DIR}/malformed_completed.obj)
set_tests_properties(complete_malformed PROPERTIES PASS_REGULAR_EXPRESSION "malformed vertex")
# On a closed sphere the winding number is 1 inside and 0 outside.
add_test(NAME slice COMMAND gsd_slice --sphere 0 ${CMAKE_CURRENT_BINARY_DIR}/sphere_slice.txt --resolution 9)
set_tests_properties(slice PROPERTIES PASS_REGULAR_EXPRESSION "w from -?0\\.000 to 1\\.000")
