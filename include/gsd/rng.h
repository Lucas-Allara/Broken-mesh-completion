// The random generator of numpy (default_rng: SeedSequence and PCG64) and the three draws that
// the pipeline uses: random, integers and choice with probabilities. With the same seed it gives
// the same numbers as numpy, so that the samples of the surfaces and the random holes are those
// of the reference values of the tests (tests/reference.h).
#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>

namespace gsd {

class Rng {
public:
    explicit Rng(std::uint64_t seed = 0);
    std::uint64_t next64();
    std::uint32_t next32();
    // Uniform in [0, 1), as Generator.random.
    double random();
    // Uniform integer in [0, high), as Generator.integers(high).
    std::uint64_t integers(std::uint64_t high);
    // k indices drawn with probabilities p (normalized here), as Generator.choice(n, size=k, p=p).
    std::vector<int> choice(const std::vector<double>& p, int k);

private:
    unsigned __int128 state_ = 0, inc_ = 0;
    bool has_uint32_ = false;
    std::uint32_t uinteger_ = 0;
    void step();
};

// Sum of an array by pairs, as numpy.sum does it, so that the totals have the same rounding.
double pairwise_sum(const double* a, std::size_t n);
inline double pairwise_sum_of(const std::vector<double>& a) { return pairwise_sum(a.data(), a.size()); }

}  // namespace gsd
