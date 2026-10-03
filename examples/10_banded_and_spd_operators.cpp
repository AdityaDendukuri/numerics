/// @file 10_banded_and_spd_operators.cpp
/// @brief Band storage, banded LU with pivoting, and solves with one or many right-hand sides.
///
/// `band_mat(n, kl, ku)` stores the kl subdiagonals and ku superdiagonals in LAPACK band
/// layout, with room for the fill that pivoting adds. `lu(band)` factors it with partial
/// pivoting in O(n kl (kl + ku)), and `solve` takes a vector or a matrix of right-hand sides.
#include <cmath>
#include <cstdio>
#include <numerics.hpp>
#include <string_view>
#include <vector>

using namespace num;

int main(int argc, char **argv) {
    const bool plot = argc > 1 && std::string_view(argv[1]) == "--plot";

    // The tridiagonal matrix T = tridiag(-1, 4, -1) with adjacent rows swapped in pairs: still
    // well conditioned, but its large entries now sit off the diagonal, so partial pivoting
    // swaps rows at every other step. The swap widens the band to two on each side.
    constexpr idx n = 2000, kl = 2, ku = 2;
    band_mat B(n, kl, ku, 0.0);
    mat<real> dense(n, n, 0.0);
    for (idx i = 0; i < n; ++i) {
        const idx row = (i ^ 1) < n ? (i ^ 1) : i; // the row of T placed at row i
        for (idx j = (row > 0 ? row - 1 : 0); j <= std::min(row + 1, n - 1); ++j) {
            const real value = (j == row) ? 4.0 : -1.0;
            B(i, j) = value;
            dense(i, j) = value;
        }
    }
    vec<real> b(n);
    for (idx i = 0; i < n; ++i) {
        b[i] = std::sin(0.01 * static_cast<real>(i));
    }

    const banded_lu_result F = lu(B);
    idx swaps = 0;
    for (idx k = 0; k < n; ++k) {
        swaps += F.swaps[k] != k ? 1 : 0;
    }
    vec<real> x, product(n);
    solve(F, b, x);
    banded_matvec(B, x, product);
    real residual = 0.0, size = 0.0;
    for (idx i = 0; i < n; ++i) {
        residual = std::max(residual, std::abs(product[i] - b[i]));
        size = std::max(size, std::abs(x[i]));
    }
    std::printf("n = %zu, kl = %zu, ku = %zu: %zu row swaps, |x| %.2f, residual %.1e\n",
                static_cast<std::size_t>(n), static_cast<std::size_t>(kl),
                static_cast<std::size_t>(ku), static_cast<std::size_t>(swaps), size, residual);

    vec<real> dense_x;
    solve(lu(dense), b, dense_x);
    real difference = 0.0;
    for (idx i = 0; i < n; ++i) {
        difference = std::max(difference, std::abs(dense_x[i] - x[i]));
    }
    std::printf("agreement with a dense LU of the same matrix: %.1e\n", difference);

    // Many right-hand sides at once, one per column.
    mat<real> rhs(n, 8, 0.0), X;
    for (idx i = 0; i < n; ++i) {
        for (idx c = 0; c < 8; ++c) {
            rhs(i, c) = b[i] * static_cast<real>(c + 1);
        }
    }
    solve(F, rhs, X);
    std::printf("8 right-hand sides: X(10, 7) / x[10] = %.6f (expect 8)\n", X(10, 7) / x[10]);

    if (plot) {
        std::vector<double> grid, sol;
        for (idx i = 0; i < n; i += 10) {
            grid.push_back(static_cast<double>(i));
            sol.push_back(x[i]);
        }
        plt::plot(grid, sol, "x", "lines");
        plt::title("10 Banded solve");
        plt::show();
    }
}
