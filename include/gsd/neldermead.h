// The Nelder-Mead simplex method, with the initial simplex, the coefficients and the stopping
// rule of scipy.optimize.minimize(method="Nelder-Mead").
#pragma once
#include <functional>
#include <vector>

namespace gsd {

struct NelderMeadResult {
    std::vector<double> x;
    double fun = 0.0;
    int iterations = 0;
};

NelderMeadResult nelder_mead(const std::function<double(const std::vector<double>&)>& f, const std::vector<double>& x0,
                             double xatol = 1e-4, double fatol = 1e-4, int maxiter = -1);

}  // namespace gsd
