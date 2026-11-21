// Minimal checks for the tests: each failed check is printed and makes the test fail.
#pragma once
#include <cmath>
#include <cstdio>

namespace check {
inline int failures = 0;
inline void that(bool ok, const char* what, const char* file, int line) {
    if (!ok) {
        std::printf("FAILED %s:%d: %s\n", file, line, what);
        failures++;
    }
}
inline bool near(double a, double b, double tol) { return std::fabs(a - b) <= tol * (1.0 + std::fabs(b)); }
inline int result(const char* name) {
    std::printf("%s: %s\n", name, failures ? "FAILED" : "ok");
    return failures ? 1 : 0;
}
}  // namespace check

#define CHECK(condition) check::that((condition), #condition, __FILE__, __LINE__)
#define CHECK_NEAR(a, b, tol) check::that(check::near((a), (b), (tol)), #a " == " #b, __FILE__, __LINE__)
