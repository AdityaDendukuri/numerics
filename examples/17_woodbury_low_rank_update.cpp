/// @file 17_woodbury_low_rank_update.cpp
/// @brief Solving with A + PQ^T from a factorization of A, by the Woodbury identity.
///
/// The Woodbury identity \f$(A + PQ^T)^{-1} = A^{-1} - A^{-1}P(I + Q^TA^{-1}P)^{-1}Q^TA^{-1}\f$
/// means a rank-p change costs p solves with the old factors and one p-by-p factorization, instead
/// of a new \f$O(n^3)\f$ one. `low_rank_difference` writes the change between two matrices that
/// differ in a few rows and columns as \f$PQ^T\f$. `woodbury_solver` accepts any factorization of
/// A, and here it is a dense LU.
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numerics.hpp>
#include <optional>
#include <random>

using namespace num;

namespace {

mat<real> diagonally_dominant(idx n, unsigned seed) {
    std::mt19937 generator(seed);
    std::uniform_real_distribution<real> entry(-1.0, 1.0);
    mat<real> A(n, n, 0.0);
    for (idx i = 0; i < n; ++i) {
        real row = 0.0;
        for (idx j = 0; j < n; ++j) {
            if (i != j) {
                A(i, j) = entry(generator) / static_cast<real>(n);
                row += std::abs(A(i, j));
            }
        }
        A(i, i) = 1.0 + row;
    }
    return A;
}

spmat to_sparse(const mat<real> &A) {
    array<idx> rows, columns;
    array<real> values;
    for (idx i = 0; i < A.rows(); ++i) {
        for (idx j = 0; j < A.cols(); ++j) {
            if (A(i, j) != 0.0) {
                rows.push_back(i);
                columns.push_back(j);
                values.push_back(A(i, j));
            }
        }
    }
    return spmat::from_triplets(A.rows(), A.cols(), rows, columns, values);
}

real max_difference(const vec<real> &x, const vec<real> &y) {
    real worst = 0.0;
    for (idx i = 0; i < x.size(); ++i) {
        worst = std::max(worst, std::abs(x[i] - y[i]));
    }
    return worst;
}

template <class Work>
double milliseconds(Work &&work) {
    const auto start = std::chrono::steady_clock::now();
    work();
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
        .count();
}

} // namespace

int main() {
    constexpr idx n = 800;
    const mat<real> A = diagonally_dominant(n, 1);
    const lu_result<real> F = lu(A); // factored once

    // Rows and columns 5, 100 and 640 change: a rank-6 update.
    const array<idx> changed{5, 100, 640};
    mat<real> B = A;
    for (idx k : changed) {
        for (idx j = 0; j < n; ++j) {
            B(k, j) *= 1.5;
            B(j, k) *= 1.5;
        }
    }

    vec<real> b(n);
    for (idx i = 0; i < n; ++i) {
        b[i] = std::sin(static_cast<real>(i));
    }

    const spmat A_sparse = to_sparse(A), B_sparse = to_sparse(B);
    std::optional<woodbury_solver<lu_result<real>>> W;
    vec<real> corrected, fresh;
    const double correct_ms = milliseconds([&] {
        W.emplace(F, low_rank_difference(A_sparse, B_sparse, changed));
        solve(*W, b, corrected);
    });
    const double fresh_ms = milliseconds([&] { solve(lu(B), b, fresh); });

    std::cout << "rank of the update: " << W->rank() << "\n" << std::fixed << std::setprecision(2)
              << "Woodbury solve: " << correct_ms << " ms, fresh LU: " << fresh_ms << " ms\n"
              << std::scientific << std::setprecision(1)
              << "|x_woodbury - x_fresh|   = " << max_difference(corrected, fresh) << "\n";

    // The corrected solver is a factorization like any other: transpose(W) solves with B^T.
    vec<real> xt_fresh;
    const vec<real> xt = solve(transpose(*W), b);
    solve_transpose(lu(B), b, xt_fresh);
    std::cout << "|x^T_woodbury - x^T_fresh| = " << max_difference(xt, xt_fresh) << "\n";

    // diag(B^{-1}) from diag(A^{-1}) and the two rank-p blocks W already holds.
    const mat<real> inverse_A = inverse(F);
    vec<real> base_diagonal(n);
    for (idx i = 0; i < n; ++i) {
        base_diagonal[i] = inverse_A(i, i);
    }
    const vec<real> diagonal = W->inverse_diagonal(base_diagonal);
    const mat<real> inverse_B = inverse(lu(B));
    real worst = 0.0;
    for (idx i = 0; i < n; ++i) {
        worst = std::max(worst, std::abs(diagonal[i] - inverse_B(i, i)));
    }
    std::cout << "|diag(B^-1) updated - exact| = " << worst << "\n";
}
