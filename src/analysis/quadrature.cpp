/// @file quadrature.cpp
#include "analysis/quadrature.hpp"
#include "analysis/types.hpp"
#include <cmath>
#include <stdexcept>
#include <vector>

#ifdef NUMERICS_HAS_OMP
    #include <omp.h>
#endif

namespace num {

real trapz(ScalarFn f, real a, real b, idx n, Backend backend) {
    real h   = (b - a) / n;
    real sum = 0.0;
#ifdef NUMERICS_HAS_OMP
    #pragma omp parallel for reduction(+ : sum) \
        schedule(static) if (backend == Backend::omp)
#endif
    for (idx i = 1; i < n; ++i)
        sum += f(a + i * h);
    return h * (0.5 * (f(a) + f(b)) + sum);
}

real simpson(ScalarFn f, real a, real b, idx n, Backend backend) {
    if (n % 2 != 0)
        throw std::invalid_argument("simpson: n must be even");
    real h   = (b - a) / n;
    real sum = f(a) + f(b);
#ifdef NUMERICS_HAS_OMP
    #pragma omp parallel for reduction(+ : sum) \
        schedule(static) if (backend == Backend::omp)
#endif
    for (idx i = 1; i < n; ++i)
        sum += f(a + i * h) * (i % 2 == 0 ? 2.0 : 4.0);
    return h / 3.0 * sum;
}

// Gauss-Legendre nodes and weights on [-1, 1] for p = 1..5
// Source: Abramowitz & Stegun, Table 25.4
static constexpr real GL_NODES[5][5] = {
    {0.0, 0, 0, 0, 0},
    {-0.5773502691896257, 0.5773502691896257, 0, 0, 0},
    {-0.7745966692414834, 0.0, 0.7745966692414834, 0, 0},
    {-0.8611363115940526,
     -0.3399810435848563,
     0.3399810435848563,
     0.8611363115940526,
     0},
    {-0.9061798459386640,
     -0.5384693101056831,
     0.0,
     0.5384693101056831,
     0.9061798459386640}};
static constexpr real GL_WEIGHTS[5][5] = {
    {2.0, 0, 0, 0, 0},
    {1.0, 1.0, 0, 0, 0},
    {0.5555555555555556, 0.8888888888888889, 0.5555555555555556, 0, 0},
    {0.3478548451374538,
     0.6521451548625461,
     0.6521451548625461,
     0.3478548451374538,
     0},
    {0.2369268850561891,
     0.4786286704993665,
     0.5688888888888889,
     0.4786286704993665,
     0.2369268850561891}};

real gauss_legendre(ScalarFn f, real a, real b, idx p) {
    if (p < 1 || p > 5)
        throw std::invalid_argument("gauss_legendre: p must be 1..5");
    real mid  = 0.5 * (a + b);
    real half = 0.5 * (b - a);
    real sum  = 0.0;
    for (idx i = 0; i < p; ++i)
        sum += GL_WEIGHTS[p - 1][i] * f(mid + half * GL_NODES[p - 1][i]);
    return half * sum;
}

// Recursive adaptive Simpson helper
static real adaptive_helper(ScalarFn f,
                            real     a,
                            real     b,
                            real     fa,
                            real     fm,
                            real     fb,
                            real     whole,
                            real     tol,
                            idx      depth) {
    real mid_l = 0.5 * (a + b * 0.5 + a * 0.5); // midpoint of [a, mid]
    real mid_r = 0.5 * (0.5 * (a + b) + b);     // midpoint of [mid, b]
    real mid   = 0.5 * (a + b);
    real fl    = f(0.5 * (a + mid));
    real fr    = f(0.5 * (mid + b));
    (void)mid_l;
    (void)mid_r;

    real left  = (b - a) / 12.0 * (fa + 4.0 * fl + fm);
    real right = (b - a) / 12.0 * (fm + 4.0 * fr + fb);
    real delta = left + right - whole;

    if (depth == 0 || std::abs(delta) <= 15.0 * tol)
        return left + right + delta / 15.0;

    return adaptive_helper(f, a, mid, fa, fl, fm, left, tol * 0.5, depth - 1)
           + adaptive_helper(f,
                             mid,
                             b,
                             fm,
                             fr,
                             fb,
                             right,
                             tol * 0.5,
                             depth - 1);
}

real adaptive_simpson(ScalarFn f, real a, real b, real tol, idx max_depth) {
    real fa = f(a), fb = f(b), fm = f(0.5 * (a + b));
    real est = (b - a) / 6.0 * (fa + 4.0 * fm + fb);
    return adaptive_helper(f, a, b, fa, fm, fb, est, tol, max_depth);
}

real romberg(ScalarFn f, real a, real b, real tol, idx max_levels) {
    std::vector<std::vector<real>> R(max_levels,
                                     std::vector<real>(max_levels, 0.0));
    R[0][0] = 0.5 * (b - a) * (f(a) + f(b));

    for (idx i = 1; i < max_levels; ++i) {
        idx  n   = idx(1) << i; // 2^i panels
        real h   = (b - a) / n;
        real sum = 0.0;
        for (idx k = 1; k < n; k += 2)
            sum += f(a + k * h);
        R[i][0] = 0.5 * R[i - 1][0] + h * sum;

        // Richardson extrapolation
        real factor = 1.0;
        for (idx j = 1; j <= i; ++j) {
            factor *= 4.0;
            R[i][j] = R[i][j - 1]
                      + (R[i][j - 1] - R[i - 1][j - 1]) / (factor - 1.0);
        }

        if (i > 0 && std::abs(R[i][i] - R[i - 1][i - 1]) < tol)
            return R[i][i];
    }
    return R[max_levels - 1][max_levels - 1];
}

} // namespace num
