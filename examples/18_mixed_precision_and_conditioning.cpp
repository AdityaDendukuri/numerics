/// @file 18_mixed_precision_and_conditioning.cpp
/// @brief Condition estimates from any factorization, and LU in float refined to double.
///
/// `rcond(F, A)` estimates \f$1/(\|A\|_1\|A^{-1}\|_1)\f$ with a handful of solves against an
/// existing factorization. `lu(A, mixed_precision)` factors in `float`, which is about twice as
/// fast, and refines each solve in `double`. It uses the same estimate to decide when \f$A\f$ is
/// too ill-conditioned for single precision, and then factors in `double` instead.
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numerics.hpp>
#include <optional>
#include <random>

using namespace num;

namespace {

mat<real> well_conditioned(idx n) {
    std::mt19937 generator(3);
    std::uniform_real_distribution<real> entry(-1.0, 1.0);
    mat<real> A(n, n, 0.0);
    for (idx i = 0; i < n; ++i) {
        for (idx j = 0; j < n; ++j) {
            A(i, j) = entry(generator);
        }
        A(i, i) += 2.0 * std::sqrt(static_cast<real>(n));
    }
    return A;
}

mat<real> hilbert(idx n) {
    mat<real> H(n, n, 0.0);
    for (idx i = 0; i < n; ++i) {
        for (idx j = 0; j < n; ++j) {
            H(i, j) = 1.0 / static_cast<real>(i + j + 1);
        }
    }
    return H;
}

// ||b - A x||_inf / (||A||_inf ||x||_inf): about machine epsilon for a backward-stable solve.
real backward_error(const mat<real> &A, const vec<real> &b, const vec<real> &x) {
    real residual = 0.0, a_norm = 0.0, x_norm = 0.0;
    for (idx i = 0; i < A.rows(); ++i) {
        real product = 0.0, row = 0.0;
        for (idx j = 0; j < A.cols(); ++j) {
            product += A(i, j) * x[j];
            row += std::abs(A(i, j));
        }
        residual = std::max(residual, std::abs(b[i] - product));
        a_norm = std::max(a_norm, row);
        x_norm = std::max(x_norm, std::abs(x[i]));
    }
    return residual / (a_norm * x_norm);
}

template <class Work>
double milliseconds(Work &&work) {
    const auto start = std::chrono::steady_clock::now();
    work();
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
        .count();
}

void report(const char *name, const mat<real> &A) {
    const vec<real> b(A.rows(), 1.0);
    vec<real> x_double, x_mixed;
    lu_result<real> D;
    std::optional<mixed_lu> M;
    const double double_ms = milliseconds([&] {
        D = seq::lu(A);
        solve(D, b, x_double);
    });
    const double mixed_ms = milliseconds([&] {
        M.emplace(lu(A, mixed_precision));
        solve(*M, b, x_mixed);
    });

    std::cout << name << " (n = " << A.rows() << ")\n"
              << std::scientific << std::setprecision(1)
              << "  rcond                " << rcond(D, A) << "\n"
              << "  mixed precision      "
              << (M->refined() ? "refined from float" : "fell back to double") << "\n"
              << "  backward error       double " << backward_error(A, b, x_double) << ", mixed "
              << backward_error(A, b, x_mixed) << "\n";
    if (A.rows() >= 256) {
        std::cout << std::fixed << std::setprecision(1) << "  factor + solve       double "
                  << double_ms << " ms, mixed " << mixed_ms << " ms\n";
    }
}

} // namespace

int main() {
    report("random, diagonally weighted", well_conditioned(1024));
    report("Hilbert", hilbert(12));
}
