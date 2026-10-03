/// @file 19_probed_inverse_diagonal.cpp
/// @brief diag(A^{-1}) of a sparse M-matrix from a block of random probes.
///
/// For a nonsingular M-matrix, `inverse_diagonal(F, A)` estimates every entry of
/// \f$\operatorname{diag}(A^{-1})\f$ at once from a block of Gaussian probes. It first scales
/// \f$A\f$ by a diagonal similarity that makes its symmetric part positive definite, which turns
/// each entry into a squared row norm. The estimate is unbiased and positive, and its relative
/// error shrinks like \f$\sqrt{2/\text{probes}}\f$. `F` can be any factorization of \f$A\f$. Here
/// it is the sparse one, which also gives the exact diagonal below with one solve per entry.
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numerics.hpp>

using namespace num;

namespace {

// A ring of n states with unequal clockwise and counterclockwise rates, plus decay: a
// nonsymmetric, nonreversible M-matrix.
spmat ring(idx n) {
    array<idx> rows, columns;
    array<real> values;
    for (idx i = 0; i < n; ++i) {
        const real forward = 1.0 + (0.5 * std::sin(static_cast<real>(i)));
        const real backward = 0.4;
        rows.insert(rows.end(), {i, i, i});
        columns.insert(columns.end(), {(i + 1) % n, (i + n - 1) % n, i});
        values.insert(values.end(), {-forward, -backward, forward + backward + 0.05});
    }
    return spmat::from_triplets(n, n, rows, columns, values);
}

} // namespace

int main() {
    constexpr idx n = 1000;
    const spmat A = ring(n);
    const auto F = lu(A, sparse);

    vec<real> exact(n), unit(n, 0.0), column;
    for (idx i = 0; i < n; ++i) {
        unit[i] = 1.0;
        solve(F, unit, column); // column i of A^{-1}
        exact[i] = column[i];
        unit[i] = 0.0;
    }

    std::cout << "n = " << n << "\n" << std::fixed;
    for (idx probes : {50, 200, 800}) {
        const vec<real> estimate = inverse_diagonal(F, A, {}, {.probes = probes, .seed = 7});
        real mean = 0.0, worst = 0.0;
        for (idx i = 0; i < n; ++i) {
            const real relative = std::abs(estimate[i] - exact[i]) / exact[i];
            mean += relative / static_cast<real>(n);
            worst = std::max(worst, relative);
        }
        std::cout << std::setw(4) << probes << " probes: " << std::setprecision(3)
                  << "mean relative error " << mean << ", worst " << worst
                  << ", sqrt(2/probes) = " << std::sqrt(2.0 / static_cast<real>(probes)) << "\n";
    }
}
