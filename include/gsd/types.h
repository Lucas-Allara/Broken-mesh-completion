// Basic types of the library: a 3D vector, index arrays and hashing of index pairs and triples.
#pragma once
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace gsd {

struct Vec3 {
    double x = 0.0, y = 0.0, z = 0.0;
    double& operator[](int i) { return i == 0 ? x : (i == 1 ? y : z); }
    double operator[](int i) const { return i == 0 ? x : (i == 1 ? y : z); }
};

inline Vec3 operator+(const Vec3& a, const Vec3& b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
inline Vec3 operator-(const Vec3& a, const Vec3& b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
inline Vec3 operator-(const Vec3& a) { return {-a.x, -a.y, -a.z}; }
inline Vec3 operator*(const Vec3& a, double s) { return {a.x * s, a.y * s, a.z * s}; }
inline Vec3 operator*(double s, const Vec3& a) { return {a.x * s, a.y * s, a.z * s}; }
inline Vec3 operator/(const Vec3& a, double s) { return {a.x / s, a.y / s, a.z / s}; }
inline Vec3& operator+=(Vec3& a, const Vec3& b) { a.x += b.x; a.y += b.y; a.z += b.z; return a; }
inline Vec3& operator-=(Vec3& a, const Vec3& b) { a.x -= b.x; a.y -= b.y; a.z -= b.z; return a; }
inline double dot(const Vec3& a, const Vec3& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline Vec3 cross(const Vec3& a, const Vec3& b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
inline double norm(const Vec3& a) { return std::sqrt(dot(a, a)); }
inline Vec3 min(const Vec3& a, const Vec3& b) { return {std::fmin(a.x, b.x), std::fmin(a.y, b.y), std::fmin(a.z, b.z)}; }
inline Vec3 max(const Vec3& a, const Vec3& b) { return {std::fmax(a.x, b.x), std::fmax(a.y, b.y), std::fmax(a.z, b.z)}; }

using Tri = std::array<int, 3>;
using Tet = std::array<int, 4>;
using Edge = std::array<int, 2>;
using Points = std::vector<Vec3>;

inline std::uint64_t pair_key(int a, int b) {
    if (a > b) { const int t = a; a = b; b = t; }
    return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(a)) << 32) | static_cast<std::uint32_t>(b);
}

// Sorted copy of a triple, the key of a face whatever its orientation.
inline Tri sorted(Tri t) {
    if (t[0] > t[1]) { const int s = t[0]; t[0] = t[1]; t[1] = s; }
    if (t[1] > t[2]) { const int s = t[1]; t[1] = t[2]; t[2] = s; }
    if (t[0] > t[1]) { const int s = t[0]; t[0] = t[1]; t[1] = s; }
    return t;
}

struct TriHash {
    std::size_t operator()(const Tri& t) const {
        std::uint64_t h = 1469598103934665603ull;
        for (int v : t) { h ^= static_cast<std::uint32_t>(v); h *= 1099511628211ull; }
        return static_cast<std::size_t>(h);
    }
};

// Bounding box of a set of points.
inline void bounding_box(const Points& P, Vec3& lo, Vec3& hi) {
    lo = {INFINITY, INFINITY, INFINITY};
    hi = {-INFINITY, -INFINITY, -INFINITY};
    for (const Vec3& p : P) { lo = min(lo, p); hi = max(hi, p); }
}

}  // namespace gsd
