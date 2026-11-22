/// @file test_analysis.cpp
/// @brief Tests for root finding and quadrature
#include "quadrature/quadrature.hpp"
#include "roots/roots.hpp"
#include "core/policy.hpp"
#include <cmath>
#include <gtest/gtest.h>

using namespace num;

// Root finding

TEST(Roots, BisectionSimple) {
    // x^2 - 2 = 0, root at sqrt(2)
    auto f = [](real x) { return (x * x) - 2.0; };
    root_result<real> r = bisection(f, 1.0, 2.0);
    EXPECT_TRUE(r.converged);
    EXPECT_NEAR(r.root, std::sqrt(2.0), 1e-9);
}

TEST(Roots, BisectionBadBracketThrows) {
    auto f = [](real x) { return (x * x) + 1.0; }; // no real root
    EXPECT_THROW(bisection(f, -2.0, 2.0), std::invalid_argument);
}

TEST(Roots, NewtonQuadratic) {
    auto f = [](real x) { return (x * x) - 2.0; };
    auto df = [](real x) { return 2.0 * x; };
    root_result<real> r = newton(f, df, 1.5);
    EXPECT_TRUE(r.converged);
    EXPECT_NEAR(r.root, std::sqrt(2.0), 1e-10);
    EXPECT_LT(r.iterations, 20u);
}

TEST(Roots, NewtonTrigonometric) {
    // cos(x) = x  =>  cos(x) - x = 0, Dottie number ~0.7390851332
    auto f = [](real x) { return std::cos(x) - x; };
    auto df = [](real x) { return -std::sin(x) - 1.0; };
    root_result<real> r = newton(f, df, 0.5);
    EXPECT_TRUE(r.converged);
    EXPECT_NEAR(r.root, 0.7390851332151607, 1e-10);
}

TEST(Roots, SecantSimple) {
    auto f = [](real x) { return (x * x) - 2.0; };
    root_result<real> r = secant(f, 1.0, 2.0);
    EXPECT_TRUE(r.converged);
    EXPECT_NEAR(r.root, std::sqrt(2.0), 1e-9);
}

TEST(Roots, BrentSimple) {
    auto f = [](real x) { return (x * x) - 2.0; };
    root_result<real> r = brent(f, 1.0, 2.0);
    EXPECT_TRUE(r.converged);
    EXPECT_NEAR(r.root, std::sqrt(2.0), 1e-10);
}

TEST(Roots, BrentTrigonometric) {
    auto f = [](real x) { return std::cos(x) - x; };
    root_result<real> r = brent(f, 0.0, 1.0);
    EXPECT_TRUE(r.converged);
    EXPECT_NEAR(r.root, 0.7390851332151607, 1e-10);
}

TEST(Roots, BrentBadBracketThrows) {
    auto f = [](real x) { return (x * x) + 1.0; };
    EXPECT_THROW(brent(f, -2.0, 2.0), std::invalid_argument);
}

TEST(Roots, AllMethodsAgree) {
    // x^3 - x - 1 = 0, root ~1.3247179572
    auto f = [](real x) { return (x * x * x) - x - 1.0; };
    auto df = [](real x) { return (3.0 * x * x) - 1.0; };
    root_result<real> rb = bisection(f, 1.0, 2.0);
    root_result<real> rn = newton(f, df, 1.5);
    root_result<real> rs = secant(f, 1.0, 2.0);
    root_result<real> rr = brent(f, 1.0, 2.0);
    EXPECT_NEAR(rb.root, rr.root, 1e-8);
    EXPECT_NEAR(rn.root, rr.root, 1e-8);
    EXPECT_NEAR(rs.root, rr.root, 1e-8);
}

// Quadrature

TEST(Quadrature, TrapzPolynomial) {
    // integral of x^2 from 0 to 1 = 1/3
    auto f = [](real x) { return x * x; };
    real result = trapz(f, 0.0, 1.0, 10000);
    EXPECT_NEAR(result, 1.0 / 3.0, 1e-6);
}

TEST(Quadrature, SimpsonPolynomial) {
    // Simpson is exact for polynomials up to degree 3
    auto f = [](real x) { return x * x * x; }; // integral = 1/4
    real result = simpson(f, 0.0, 1.0, 100);
    EXPECT_NEAR(result, 0.25, 1e-10);
}

TEST(Quadrature, SimpsonOddNThrows) {
    auto f = [](real x) { return x; };
    EXPECT_THROW(simpson(f, 0.0, 1.0, 3), std::invalid_argument);
}

TEST(Quadrature, GaussLegendreTrig) {
    // integral of sin(x) from 0 to pi = 2
    auto f = [](real x) { return std::sin(x); };
    real result = gauss_legendre(f, 0.0, num::pi, 5);
    EXPECT_NEAR(result, 2.0, 1e-6); // GL accuracy on non-polynomial over [0,pi]
}

TEST(Quadrature, GaussLegendreExactPolynomial) {
    // p=3 is exact for degree <= 5; x^5 integral from 0 to 1 = 1/6
    auto f = [](real x) { return x * x * x * x * x; };
    real result = gauss_legendre(f, 0.0, 1.0, 3);
    EXPECT_NEAR(result, 1.0 / 6.0, 1e-10);
}

TEST(Quadrature, AdaptiveSimpsonTrig) {
    auto f = [](real x) { return std::sin(x); };
    real result = adaptive_simpson(f, 0.0, num::pi, {.rtol = 1e-10});
    EXPECT_NEAR(result, 2.0, 1e-9);
}

TEST(Quadrature, AdaptiveSimpsonExp) {
    // integral of e^x from 0 to 1 = e - 1
    auto f = [](real x) { return std::exp(x); };
    real result = adaptive_simpson(f, 0.0, 1.0, {.rtol = 1e-10});
    EXPECT_NEAR(result, std::exp(1.0) - 1.0, 1e-9);
}

TEST(Quadrature, RombergHighAccuracy) {
    // Romberg should achieve very high accuracy with few evaluations
    auto f = [](real x) { return std::exp(-x * x); };
    // integral from 0 to 1 (no closed form, use reference value)
    real result = romberg(f, 0.0, 1.0, {.rtol = 1e-12});
    EXPECT_NEAR(result, 0.7468241328124270, 1e-10);
}

TEST(Quadrature, AllMethodsAgree) {
    auto f = [](real x) { return std::cos(x); }; // integral from 0 to pi/2 = 1
    real t = trapz(f, 0.0, num::pi / 2.0, 10000);
    real s = simpson(f, 0.0, num::pi / 2.0, 1000);
    real g = gauss_legendre(f, 0.0, num::pi / 2.0, 5);
    real a = adaptive_simpson(f, 0.0, num::pi / 2.0);
    real r = romberg(f, 0.0, num::pi / 2.0);
    EXPECT_NEAR(t, 1.0, 1e-5);
    EXPECT_NEAR(s, 1.0, 1e-8);
    EXPECT_NEAR(g, 1.0, 1e-10);
    EXPECT_NEAR(a, 1.0, 1e-8);
    EXPECT_NEAR(r, 1.0, 1e-10);
}

// backend::omp variants must produce the same result as the sequential version.
// (If OMP is not available the omp backend falls back to serial internally.)

TEST(Quadrature, TrapzOmpMatchesSeq) {
    auto f = [](real x) { return x * x; };
    real seq_result = trapz(f, 0.0, 1.0, 10000);
    real omp_result = trapz<true>(f, 0.0, 1.0, 10000);
    EXPECT_NEAR(omp_result, seq_result, 1e-12);
}

TEST(Quadrature, SimpsonOmpMatchesSeq) {
    auto f = [](real x) { return x * x * x; };
    real seq_result = simpson(f, 0.0, 1.0, 1000);
    real omp_result = simpson<true>(f, 0.0, 1.0, 1000);
    EXPECT_NEAR(omp_result, seq_result, 1e-12);
}

// Newton on x^3 - 2x + 2 from 0 cycles 0 -> 1 -> 0 forever; it must report failure.
TEST(Roots, NewtonReportsTheTwoCycle) {
    const auto r = newton([](real x) { return x * x * x - 2 * x + 2; },
                          [](real x) { return 3 * x * x - 2; }, 0.0, {.atol = 1e-12}, 100);
    EXPECT_FALSE(r.converged) << r;
}

// Wallis's cubic x^3 - 2x - 5, the historical test problem for Newton's method.
TEST(Roots, WallisCubic) {
    const real root = 2.0945514815423265;
    auto f = [](real x) { return x * x * x - 2 * x - 5; };
    EXPECT_NEAR(newton(f, [](real x) { return 3 * x * x - 2; }, 2.0).root, root, 1e-12);
    EXPECT_NEAR(secant(f, 2.0, 3.0).root, root, 1e-12);
    EXPECT_NEAR(brent(f, 2.0, 3.0, {.atol = 1e-14}).root, root, 1e-12);
    EXPECT_NEAR(bisection(f, 2.0, 3.0, {.atol = 1e-14}).root, root, 1e-12);
}

// A residual test |f(x)| < tol is not scale invariant: multiplying f by 1e-12 does not move the
// root, but makes every point of the bracket pass an absolute residual test.
TEST(Roots, InvariantToScalingF) {
    auto f = [](real x) { return 1e-12 * (x - pi); };
    auto df = [](real) { return 1e-12; };
    EXPECT_NEAR(bisection(f, 0.0, 4.0).root, pi, 1e-8);
    EXPECT_NEAR(brent(f, 0.0, 4.0).root, pi, 1e-8);
    EXPECT_NEAR(newton(f, df, 0.0).root, pi, 1e-8);
    EXPECT_NEAR(secant(f, 0.0, 4.0).root, pi, 1e-8);
}

// With an absolute step tolerance, a root near 1e8 can only be located to the spacing of
// doubles there (1.5e-8); rtol asks for relative accuracy instead.
TEST(Roots, AcceptARelativeStepTolerance) {
    const real root = 1.2345e8;
    auto f = [root](real x) { return (x - root) * (x + 1.0); };
    const auto r = brent(f, 1e8, 2e8, {.rtol = 1e-14});
    EXPECT_TRUE(r.converged) << r;
    EXPECT_LE(std::abs(r.root - root), 1e-14 * root * 2);
    const auto b = bisection(f, 1e8, 2e8, {.atol = 1e-12});
    EXPECT_TRUE(b.converged) << b; // floored at 4 eps |x| rather than spinning to max_iter
}

// The trapezoid rule converges geometrically on smooth periodic integrands (Trefethen and
// Weideman): 16 points already give 2*pi*I_0(1) to machine precision.
TEST(Quadrature, TrapezoidIsSpectrallyAccurateOnPeriodicIntegrands) {
    const real exact = 2.0 * num::pi * 1.2660658777520082;
    const real got = trapz([](real x) { return std::exp(std::cos(x)); }, 0.0, 2.0 * num::pi, 16);
    EXPECT_NEAR(got, exact, 4.0 * eps * exact);
}

// An endpoint singularity in the derivative breaks the error expansion of Simpson's rule;
// adaptivity must still reach the requested tolerance.
TEST(Quadrature, AdaptiveSimpsonHandlesSqrtEndpointSingularity) {
    EXPECT_NEAR(adaptive_simpson([](real x) { return std::sqrt(x); }, 0.0, 1.0, {.rtol = 1e-10}), 2.0 / 3.0, 1e-9);
    EXPECT_NEAR(romberg([](real x) { return std::abs(x - 1.0 / 3.0); }, 0.0, 1.0, {.rtol = 1e-10}), 5.0 / 18.0, 1e-6);
}

// The scale of an integral is the integral of |f|, so 1e-12 * f is integrated to the same
// relative accuracy as f, and an integral that cancels to zero still terminates.
TEST(Quadrature, ToleranceIsRelativeToTheIntegrand) {
    for (real s : {1.0, 1e-12, 1e12}) {
        auto f = [s](real x) { return s * std::sqrt(x); };
        EXPECT_NEAR(adaptive_simpson(f, 0.0, 1.0) / s, 2.0 / 3.0, 1e-9) << "s = " << s;
        auto g = [s](real x) { return s * std::exp(x); };
        EXPECT_NEAR(romberg(g, 0.0, 1.0) / s, num::e - 1.0, 1e-9) << "s = " << s;
    }
    EXPECT_NEAR(adaptive_simpson([](real x) { return std::sin(x); }, 0.0, num::two_pi), 0.0, 1e-9);
    // Zero at all three Simpson nodes: a scale sampled there alone would be 0, leaving no bound.
    EXPECT_NEAR(adaptive_simpson([](real x) { return std::sin(num::two_pi * x) * std::sin(num::two_pi * x); },
                                 0.0, 1.0),
                0.5, 1e-9);
}
