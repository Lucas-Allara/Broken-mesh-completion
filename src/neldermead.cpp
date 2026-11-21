#include "gsd/neldermead.h"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace gsd {

NelderMeadResult nelder_mead(const std::function<double(const std::vector<double>&)>& f, const std::vector<double>& x0,
                             double xatol, double fatol, int maxiter) {
    const double rho = 1.0, chi = 2.0, psi = 0.5, sigma = 0.5, nonzdelt = 0.05, zdelt = 0.00025;
    const int N = static_cast<int>(x0.size());
    if (maxiter < 0) maxiter = 200 * N;

    std::vector<std::vector<double>> sim(N + 1, x0);
    for (int k = 0; k < N; k++) sim[k + 1][k] = x0[k] != 0 ? (1 + nonzdelt) * x0[k] : zdelt;
    std::vector<double> fsim(N + 1);
    for (int k = 0; k <= N; k++) fsim[k] = f(sim[k]);

    auto order = [&]() {
        std::vector<int> ind(N + 1);
        std::iota(ind.begin(), ind.end(), 0);
        std::stable_sort(ind.begin(), ind.end(), [&](int a, int b) { return fsim[a] < fsim[b]; });
        std::vector<std::vector<double>> s(N + 1);
        std::vector<double> fs(N + 1);
        for (int k = 0; k <= N; k++) { s[k] = sim[ind[k]]; fs[k] = fsim[ind[k]]; }
        sim = std::move(s);
        fsim = std::move(fs);
    };
    order();

    int iterations = 1;
    std::vector<double> xbar(N), x(N);
    auto combine = [&](double a, double b) {  // a * xbar + b * sim[-1]
        for (int j = 0; j < N; j++) x[j] = a * xbar[j] + b * sim[N][j];
    };
    while (iterations < maxiter) {
        double dx = 0.0, df = 0.0;
        for (int k = 1; k <= N; k++) {
            for (int j = 0; j < N; j++) dx = std::max(dx, std::fabs(sim[k][j] - sim[0][j]));
            df = std::max(df, std::fabs(fsim[0] - fsim[k]));
        }
        if (dx <= xatol && df <= fatol) break;

        for (int j = 0; j < N; j++) {
            double s = 0.0;
            for (int k = 0; k < N; k++) s += sim[k][j];
            xbar[j] = s / N;
        }
        combine(1 + rho, -rho);
        const std::vector<double> xr = x;
        const double fxr = f(xr);
        bool shrink = false;
        if (fxr < fsim[0]) {
            combine(1 + rho * chi, -rho * chi);
            const std::vector<double> xe = x;
            const double fxe = f(xe);
            if (fxe < fxr) { sim[N] = xe; fsim[N] = fxe; }
            else { sim[N] = xr; fsim[N] = fxr; }
        } else if (fxr < fsim[N - 1]) {
            sim[N] = xr;
            fsim[N] = fxr;
        } else if (fxr < fsim[N]) {
            combine(1 + psi * rho, -psi * rho);
            const std::vector<double> xc = x;
            const double fxc = f(xc);
            if (fxc <= fxr) { sim[N] = xc; fsim[N] = fxc; }
            else shrink = true;
        } else {
            combine(1 - psi, psi);
            const std::vector<double> xcc = x;
            const double fxcc = f(xcc);
            if (fxcc < fsim[N]) { sim[N] = xcc; fsim[N] = fxcc; }
            else shrink = true;
        }
        if (shrink)
            for (int k = 1; k <= N; k++) {
                for (int j = 0; j < N; j++) sim[k][j] = sim[0][j] + sigma * (sim[k][j] - sim[0][j]);
                fsim[k] = f(sim[k]);
            }
        iterations++;
        order();
    }
    NelderMeadResult r;
    r.x = sim[0];
    r.fun = *std::min_element(fsim.begin(), fsim.end());
    r.iterations = iterations;
    return r;
}

}  // namespace gsd
