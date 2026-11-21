#include "gsd/rng.h"

#include <algorithm>
#include <array>
#include <stdexcept>

namespace gsd {
namespace {

// SeedSequence of numpy (numpy/random/bit_generator.pyx), for one 32-bit word of entropy or two.
constexpr std::uint32_t kInitA = 0x43b0d7e5u, kMultA = 0x931e8875u, kInitB = 0x8b51f9ddu, kMultB = 0x58f38dedu;
constexpr std::uint32_t kMixL = 0xca01f9ddu, kMixR = 0x4973f715u;
constexpr int kShift = 16, kPool = 4;

std::uint32_t hashmix(std::uint32_t value, std::uint32_t& hash_const) {
    value ^= hash_const;
    hash_const *= kMultA;
    value *= hash_const;
    value ^= value >> kShift;
    return value;
}

std::uint32_t mix(std::uint32_t x, std::uint32_t y) {
    std::uint32_t r = kMixL * x - kMixR * y;
    r ^= r >> kShift;
    return r;
}

// The four 64-bit words that PCG64 takes from the seed sequence.
std::array<std::uint64_t, 4> seed_words(std::uint64_t seed) {
    std::vector<std::uint32_t> entropy;
    if (seed == 0) entropy.push_back(0);
    for (std::uint64_t n = seed; n > 0; n >>= 32) entropy.push_back(static_cast<std::uint32_t>(n & 0xffffffffu));
    std::uint32_t pool[kPool];
    std::uint32_t hash_const = kInitA;
    for (int i = 0; i < kPool; i++) pool[i] = hashmix(i < static_cast<int>(entropy.size()) ? entropy[i] : 0u, hash_const);
    for (int src = 0; src < kPool; src++)
        for (int dst = 0; dst < kPool; dst++)
            if (src != dst) pool[dst] = mix(pool[dst], hashmix(pool[src], hash_const));
    for (std::size_t src = kPool; src < entropy.size(); src++)
        for (int dst = 0; dst < kPool; dst++) pool[dst] = mix(pool[dst], hashmix(entropy[src], hash_const));

    std::uint32_t words[8];
    std::uint32_t c = kInitB;
    for (int i = 0; i < 8; i++) {
        std::uint32_t v = pool[i % kPool];
        v ^= c;
        c *= kMultB;
        v *= c;
        v ^= v >> kShift;
        words[i] = v;
    }
    std::array<std::uint64_t, 4> out;
    for (int i = 0; i < 4; i++) out[i] = static_cast<std::uint64_t>(words[2 * i]) | (static_cast<std::uint64_t>(words[2 * i + 1]) << 32);
    return out;
}

constexpr unsigned __int128 kMultiplier =
    (static_cast<unsigned __int128>(2549297995355413924ull) << 64) + 4865540595714422341ull;

}  // namespace

Rng::Rng(std::uint64_t seed) {
    const std::array<std::uint64_t, 4> w = seed_words(seed);
    const unsigned __int128 initstate = (static_cast<unsigned __int128>(w[0]) << 64) + w[1];
    const unsigned __int128 initseq = (static_cast<unsigned __int128>(w[2]) << 64) + w[3];
    state_ = 0;
    inc_ = (initseq << 1) | 1u;
    step();
    state_ += initstate;
    step();
}

void Rng::step() { state_ = state_ * kMultiplier + inc_; }

std::uint64_t Rng::next64() {
    step();
    const std::uint64_t value = static_cast<std::uint64_t>(state_ >> 64) ^ static_cast<std::uint64_t>(state_);
    const unsigned rot = static_cast<unsigned>(state_ >> 122);
    return (value >> rot) | (value << ((-rot) & 63u));
}

std::uint32_t Rng::next32() {
    if (has_uint32_) {
        has_uint32_ = false;
        return uinteger_;
    }
    const std::uint64_t next = next64();
    has_uint32_ = true;
    uinteger_ = static_cast<std::uint32_t>(next >> 32);
    return static_cast<std::uint32_t>(next & 0xffffffffu);
}

double Rng::random() { return static_cast<double>(next64() >> 11) * (1.0 / 9007199254740992.0); }

std::uint64_t Rng::integers(std::uint64_t high) {
    const std::uint64_t rng = high - 1;  // inclusive range
    if (rng == 0) return 0;
    if (rng <= 0xffffffffull) {
        if (rng == 0xffffffffull) return next32();
        // Lemire's method on 32 bits, with rejection.
        const std::uint32_t excl = static_cast<std::uint32_t>(rng) + 1u;
        std::uint64_t m = static_cast<std::uint64_t>(next32()) * excl;
        std::uint32_t leftover = static_cast<std::uint32_t>(m & 0xffffffffu);
        if (leftover < excl) {
            const std::uint32_t threshold = (0xffffffffu - static_cast<std::uint32_t>(rng)) % excl;
            while (leftover < threshold) {
                m = static_cast<std::uint64_t>(next32()) * excl;
                leftover = static_cast<std::uint32_t>(m & 0xffffffffu);
            }
        }
        return m >> 32;
    }
    if (rng == 0xffffffffffffffffull) return next64();
    // Lemire's method on 64 bits.
    const std::uint64_t excl = rng + 1u;
    unsigned __int128 m = static_cast<unsigned __int128>(next64()) * excl;
    std::uint64_t leftover = static_cast<std::uint64_t>(m);
    if (leftover < excl) {
        const std::uint64_t threshold = (0xffffffffffffffffull - rng) % excl;
        while (leftover < threshold) {
            m = static_cast<unsigned __int128>(next64()) * excl;
            leftover = static_cast<std::uint64_t>(m);
        }
    }
    return static_cast<std::uint64_t>(m >> 64);
}

std::vector<int> Rng::choice(const std::vector<double>& p, int k) {
    if (p.empty()) throw std::invalid_argument("choice needs at least one probability");
    // cdf = p.cumsum(); cdf /= cdf[-1]; searchsorted(uniform, side="right").
    std::vector<double> cdf(p.size());
    double acc = 0.0;
    for (std::size_t i = 0; i < p.size(); i++) { acc += p[i]; cdf[i] = acc; }
    const double total = cdf.back();
    for (double& c : cdf) c /= total;
    std::vector<int> out(k);
    for (int i = 0; i < k; i++) {
        const double u = random();
        out[i] = static_cast<int>(std::upper_bound(cdf.begin(), cdf.end(), u) - cdf.begin());
    }
    return out;
}

double pairwise_sum(const double* a, std::size_t n) {
    if (n < 8) {
        double res = 0.0;
        for (std::size_t i = 0; i < n; i++) res += a[i];
        return res;
    }
    if (n <= 128) {
        double r[8];
        for (int j = 0; j < 8; j++) r[j] = a[j];
        std::size_t i;
        for (i = 8; i < n - (n % 8); i += 8)
            for (int j = 0; j < 8; j++) r[j] += a[i + j];
        double res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        for (; i < n; i++) res += a[i];
        return res;
    }
    std::size_t n2 = n / 2;
    n2 -= n2 % 8;
    return pairwise_sum(a, n2) + pairwise_sum(a + n2, n - n2);
}

}  // namespace gsd
