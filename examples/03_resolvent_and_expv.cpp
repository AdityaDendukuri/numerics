/// @file 03_resolvent_and_expv.cpp
/// @brief Shifted solves (sI - A)x = b at many shifts, and the action of the matrix exponential.
///
/// `hessenberg_resolvent R(A)` reduces A to Hessenberg form once. After that, `shift(R, s)` factors
/// sI - A in O(n^2), and `solve_batch` solves at many shifts in parallel. `expv(t, A, v)` computes
/// \f$e^{tA}v\f$ from a Krylov subspace without ever forming \f$e^{tA}\f$. When A is a Markov
/// generator, \f$e^{tA}\f$ conserves total probability, which the output shows.
#include <cmath>
#include <complex>
#include <cstdio>
#include <numerics.hpp>
#include <string_view>
#include <vector>

using namespace num;

int main(int argc, char **argv) {
    const bool plot = argc > 1 && std::string_view(argv[1]) == "--plot";

    // A birth-death chain on 30 states: A is a generator (columns sum to zero).
    constexpr idx n = 30;
    mat<real> A(n, n, 0.0);
    for (idx j = 0; j < n; ++j) {
        if (j + 1 < n) {
            A(j + 1, j) += 1.0; // birth
            A(j, j) -= 1.0;
        }
        if (j > 0) {
            A(j - 1, j) += 0.6; // death
            A(j, j) -= 0.6;
        }
    }
    vec<real> p0(n, 0.0);
    p0[0] = 1.0;

    // One shift, checked against its residual.
    const hessenberg_resolvent R(A);
    const cplx s(1.0, 2.0);
    const vec<cplx> z = solve(shift(R, s), p0);
    real residual = 0.0;
    for (idx i = 0; i < n; ++i) {
        cplx row = s * z[i];
        for (idx j = 0; j < n; ++j) {
            row -= A(i, j) * z[j];
        }
        residual = std::max(residual, std::abs(row - p0[i]));
    }
    std::printf("(sI - A)z = p0 at s = 1+2i:  z[0] = %.5f%+.5fi  residual %.1e\n", z[0].real(),
                z[0].imag(), residual);

    // Many shifts from the one reduction.
    array<cplx> shifts;
    for (int k = 0; k < 64; ++k) {
        shifts.push_back(cplx(1.0, 0.25 * k));
    }
    const auto batch = solve_batch(R, shifts, p0);
    std::printf("%zu shifts solved from one Hessenberg reduction; |z[0]| at s = 1+15.75i: %.5f\n",
                batch.size(), std::abs(batch.back()[0]));

    // p(t) = e^{tA} p0 conserves probability.
    const operators::dense_op op(A);
    std::vector<double> times, mean;
    std::printf("\n   t    sum p(t)    mean state\n");
    for (real t : {0.5, 1.0, 2.0, 5.0, 10.0}) {
        const vec<real> p = expv(t, op, p0, 30, 1e-10);
        real total = 0.0, average = 0.0;
        for (idx i = 0; i < n; ++i) {
            total += p[i];
            average += static_cast<real>(i) * p[i];
        }
        std::printf("%5.1f  %10.8f  %9.4f\n", t, total, average);
        times.push_back(t);
        mean.push_back(average);
    }

    if (plot) {
        plt::plot(times, mean, "mean state", "linespoints");
        plt::title("03 Expected state of a birth-death chain");
        plt::show();
    }
}
