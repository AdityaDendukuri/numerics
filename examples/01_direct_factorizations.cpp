/// @file 01_direct_factorizations.cpp
/// @brief LU, Cholesky, QR least squares and the Thomas algorithm, all solved through solve(F, b, x).
///
/// Every factorization is a value, and the same free function solves with all of them. `lu(A)`
/// pivots, while `lu(A, no_pivot)` skips the pivot search for matrices whose structure keeps the
/// pivots nonzero. `cholesky` requires a matrix that carries the SPD law. `qr` solves least-squares
/// problems with more rows than columns, and `thomas` solves a tridiagonal system in O(n).
#include <cstdio>
#include <numerics.hpp>
#include <string_view>

using namespace num;

int main(int argc, char **argv) {
    const bool plot = argc > 1 && std::string_view(argv[1]) == "--plot";

    mat<real> A(3, 3, 0.0);
    A(0, 0) = 4.0; A(0, 1) = 1.0;
    A(1, 0) = 1.0; A(1, 1) = 4.0; A(1, 2) = 1.0;
    A(2, 1) = 1.0; A(2, 2) = 4.0;
    const vec<real> b{5.0, 6.0, 5.0};

    const auto F = lu(A);
    vec<real> x;
    solve(F, b, x);
    std::printf("LU            x = [%g, %g, %g]  det %g  rcond %.3f\n", x[0], x[1], x[2], det(F),
                rcond(F, A));
    const mat<real> I = inverse(F);
    std::printf("A^-1 (0, 0)   %.6f  (exact 15/56 = %.6f)\n", I(0, 0), 15.0 / 56.0);

    solve(lu(A, no_pivot), b, x);
    std::printf("no-pivot LU   x = [%g, %g, %g]\n", x[0], x[1], x[2]);

    solve(cholesky(assume_spd(A)), b, x);
    std::printf("Cholesky      x = [%g, %g, %g]\n", x[0], x[1], x[2]);

    vec<real> lower{1.0, 1.0}, diag{4.0, 4.0, 4.0}, upper{1.0, 1.0};
    thomas(lower, diag, upper, b, x);
    std::printf("Thomas        x = [%g, %g, %g]\n", x[0], x[1], x[2]);

    // Least squares: the line c0 + c1 t through five noisy points.
    const real t[5] = {0.0, 1.0, 2.0, 3.0, 4.0};
    const vec<real> data{1.1, 2.9, 5.2, 7.1, 8.8};
    mat<real> V(5, 2, 0.0);
    for (idx i = 0; i < 5; ++i) {
        V(i, 0) = 1.0;
        V(i, 1) = t[i];
    }
    vec<real> c;
    solve(qr(V), data, c);
    std::printf("QR fit        y = %.3f + %.3f t\n", c[0], c[1]);

    if (plot) {
        std::vector<double> ts(t, t + 5), ys(data.data(), data.data() + 5), fit;
        for (double ti : ts) {
            fit.push_back(c[0] + (c[1] * ti));
        }
        plt::plot(ts, ys, "data", "points");
        plt::plot(ts, fit, "least-squares line", "lines");
        plt::title("01 QR least squares");
        plt::show();
    }
}
