/// @file test_analysis.cpp
/// @brief Tests for root finding and quadrature
#include <gtest/gtest.h>
#include "analysis/roots.hpp"
#include "analysis/quadrature.hpp"
#include "core/policy.hpp"
#include <cmath>

using namespace num;

// Root finding

TEST(Roots, BisectionSimple) {
    // x^2 - 2 = 0, root at sqrt(2)
    auto f = [](real x) {
        return x * x - 2.0;
    };
    RootResult r = bisection(f, 1.0, 2.0);
    EXPECT_TRUE(r.converged);
    EXPECT_NEAR(r.root, std::sqrt(2.0), 1e-9);
}

TEST(Roots, BisectionBadBracketThrows) {
    auto f = [](real x) {
        return x * x + 1.0;
    }; // no real root
    EXPECT_THROW(bisection(f, -2.0, 2.0), std::invalid_argument);
}

TEST(Roots, NewtonQuadratic) {
    auto f = [](real x) {
        return x * x - 2.0;
    };
    auto df = [](real x) {
        return 2.0 * x;
    };
    RootResult r = newton(f, df, 1.5);
    EXPECT_TRUE(r.converged);
    EXPECT_NEAR(r.root, std::sqrt(2.0), 1e-10);
    EXPECT_LT(r.iterations, 20u);
}

TEST(Roots, NewtonTrigonometric) {
    // cos(x) = x  =>  cos(x) - x = 0, Dottie number ~0.7390851332
    auto f = [](real x) {
        return std::cos(x) - x;
    };
    auto df = [](real x) {
        return -std::sin(x) - 1.0;
    };
    RootResult r = newton(f, df, 0.5);
    EXPECT_TRUE(r.converged);
    EXPECT_NEAR(r.root, 0.7390851332151607, 1e-10);
}

TEST(Roots, SecantSimple) {
    auto f = [](real x) {
        return x * x - 2.0;
    };
    RootResult r = secant(f, 1.0, 2.0);
    EXPECT_TRUE(r.converged);
    EXPECT_NEAR(r.root, std::sqrt(2.0), 1e-9);
}

TEST(Roots, BrentSimple) {
    auto f = [](real x) {
        return x * x - 2.0;
    };
    RootResult r = brent(f, 1.0, 2.0);
    EXPECT_TRUE(r.converged);
    EXPECT_NEAR(r.root, std::sqrt(2.0), 1e-10);
}

TEST(Roots, BrentTrigonometric) {
    auto f = [](real x) {
        return std::cos(x) - x;
    };
    RootResult r = brent(f, 0.0, 1.0);
    EXPECT_TRUE(r.converged);
    EXPECT_NEAR(r.root, 0.7390851332151607, 1e-10);
}

TEST(Roots, BrentBadBracketThrows) {
    auto f = [](real x) {
        return x * x + 1.0;
    };
    EXPECT_THROW(brent(f, -2.0, 2.0), std::invalid_argument);
}

TEST(Roots, AllMethodsAgree) {
    // x^3 - x - 1 = 0, root ~1.3247179572
    auto f = [](real x) {
        return x * x * x - x - 1.0;
    };
    auto df = [](real x) {
        return 3.0 * x * x - 1.0;
    };
    RootResult rb = bisection(f, 1.0, 2.0);
    RootResult rn = newton(f, df, 1.5);
    RootResult rs = secant(f, 1.0, 2.0);
    RootResult rr = brent(f, 1.0, 2.0);
    EXPECT_NEAR(rb.root, rr.root, 1e-8);
    EXPECT_NEAR(rn.root, rr.root, 1e-8);
    EXPECT_NEAR(rs.root, rr.root, 1e-8);
}

// Quadrature

TEST(Quadrature, TrapzPolynomial) {
    // integral of x^2 from 0 to 1 = 1/3
    auto f = [](real x) {
        return x * x;
    };
    real result = trapz(f, 0.0, 1.0, 10000);
    EXPECT_NEAR(result, 1.0 / 3.0, 1e-6);
}

TEST(Quadrature, SimpsonPolynomial) {
    // Simpson is exact for polynomials up to degree 3
    auto f = [](real x) {
        return x * x * x;
    }; // integral = 1/4
    real result = simpson(f, 0.0, 1.0, 100);
    EXPECT_NEAR(result, 0.25, 1e-10);
}

TEST(Quadrature, SimpsonOddNThrows) {
    auto f = [](real x) {
        return x;
    };
    EXPECT_THROW(simpson(f, 0.0, 1.0, 3), std::invalid_argument);
}

TEST(Quadrature, GaussLegendreTrig) {
    // integral of sin(x) from 0 to pi = 2
    auto f = [](real x) {
        return std::sin(x);
    };
    real result = gauss_legendre(f, 0.0, M_PI, 5);
    EXPECT_NEAR(result, 2.0, 1e-6); // GL accuracy on non-polynomial over [0,pi]
}

TEST(Quadrature, GaussLegendreExactPolynomial) {
    // p=3 is exact for degree <= 5; x^5 integral from 0 to 1 = 1/6
    auto f = [](real x) {
        return x * x * x * x * x;
    };
    real result = gauss_legendre(f, 0.0, 1.0, 3);
    EXPECT_NEAR(result, 1.0 / 6.0, 1e-10);
}

TEST(Quadrature, AdaptiveSimpsonTrig) {
    auto f = [](real x) {
        return std::sin(x);
    };
    real result = adaptive_simpson(f, 0.0, M_PI, 1e-10);
    EXPECT_NEAR(result, 2.0, 1e-9);
}

TEST(Quadrature, AdaptiveSimpsonExp) {
    // integral of e^x from 0 to 1 = e - 1
    auto f = [](real x) {
        return std::exp(x);
    };
    real result = adaptive_simpson(f, 0.0, 1.0, 1e-10);
    EXPECT_NEAR(result, std::exp(1.0) - 1.0, 1e-9);
}

TEST(Quadrature, RombergHighAccuracy) {
    // Romberg should achieve very high accuracy with few evaluations
    auto f = [](real x) {
        return std::exp(-x * x);
    };
    // integral from 0 to 1 (no closed form, use reference value)
    real result = romberg(f, 0.0, 1.0, 1e-12);
    EXPECT_NEAR(result, 0.7468241328124270, 1e-10);
}

TEST(Quadrature, AllMethodsAgree) {
    auto f = [](real x) {
        return std::cos(x);
    }; // integral from 0 to pi/2 = 1
    real t = trapz(f, 0.0, M_PI / 2.0, 10000);
    real s = simpson(f, 0.0, M_PI / 2.0, 1000);
    real g = gauss_legendre(f, 0.0, M_PI / 2.0, 5);
    real a = adaptive_simpson(f, 0.0, M_PI / 2.0);
    real r = romberg(f, 0.0, M_PI / 2.0);
    EXPECT_NEAR(t, 1.0, 1e-5);
    EXPECT_NEAR(s, 1.0, 1e-8);
    EXPECT_NEAR(g, 1.0, 1e-10);
    EXPECT_NEAR(a, 1.0, 1e-8);
    EXPECT_NEAR(r, 1.0, 1e-10);
}

// Backend::omp variants must produce the same result as the sequential version.
// (If OMP is not available the omp backend falls back to serial internally.)

TEST(Quadrature, TrapzOmpMatchesSeq) {
    auto f = [](real x) {
        return x * x;
    };
    real seq_result = trapz(f, 0.0, 1.0, 10000);
    real omp_result = trapz(f, 0.0, 1.0, 10000, Backend::omp);
    EXPECT_NEAR(omp_result, seq_result, 1e-12);
}

TEST(Quadrature, SimpsonOmpMatchesSeq) {
    auto f = [](real x) {
        return x * x * x;
    };
    real seq_result = simpson(f, 0.0, 1.0, 1000);
    real omp_result = simpson(f, 0.0, 1.0, 1000, Backend::omp);
    EXPECT_NEAR(omp_result, seq_result, 1e-12);
}
