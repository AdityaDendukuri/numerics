/// @file lu_no_pivot.hpp
/// @brief LU factorization without row pivoting, for matrices whose structure
/// guarantees nonzero pivots (e.g. diagonally dominant M-matrices).
///
/// It saves the row search and swap when factoring every step. `singular` reports a zero pivot,
/// but small pivots go unchecked, so use `num::lu` without that guarantee.
#pragma once

#include "container/matrix.hpp"
#include "container/vector.hpp"
#include "kernel/kernel.hpp"
#include "linear/matrix_properties.hpp"
#include <stdexcept>

namespace num {

/// @brief Packed factorization \f$A = LU\f$ with an implicit unit-diagonal \f$L\f$
/// and no row pivoting.
struct no_pivot_lu {
    mat packed;
    bool singular = false;

    [[nodiscard]] idx size() const { return packed.rows(); }
};

/// @throws std::invalid_argument If `matrix` is not square.
/// @brief Factor \f$A = LU\f$ without row pivoting.
[[nodiscard]] inline no_pivot_lu factor_no_pivot(const mat &matrix) {
    if (matrix.rows() != matrix.cols()) {
        throw std::invalid_argument("factor_no_pivot: matrix must be square");
    }
    no_pivot_lu factor{matrix, false};
    factor.singular = !kernel::lu_no_pivot(factor.packed.data(), factor.packed.rows());
    return factor;
}

inline void solve(const no_pivot_lu &factor, const vec &rhs, vec &solution) {
    if (rhs.size() != factor.size())
        throw std::invalid_argument("no-pivot LU right-hand side size mismatch");
    solution = rhs;
    kernel::lu_no_pivot_solve_multiple(solution.data(), factor.packed.data(), factor.size(),
                                            1);
}

inline void solve(const no_pivot_lu &factor, const mat &rhs, mat &solution) {
    if (rhs.rows() != factor.size())
        throw std::invalid_argument("no-pivot LU right-hand side size mismatch");
    solution = rhs;
    kernel::lu_no_pivot_solve_multiple(solution.data(), factor.packed.data(), factor.size(),
                                            solution.cols());
}

inline void solve_transpose(const no_pivot_lu &factor, const vec &rhs, vec &solution) {
    if (rhs.size() != factor.size())
        throw std::invalid_argument("no-pivot LU right-hand side size mismatch");
    solution = rhs;
    kernel::lu_no_pivot_solve_transpose_multiple(solution.data(), factor.packed.data(),
                                                      factor.size(), 1);
}

inline void solve_transpose(const no_pivot_lu &factor, const mat &rhs, mat &solution) {
    if (rhs.rows() != factor.size())
        throw std::invalid_argument("no-pivot LU right-hand side size mismatch");
    solution = rhs;
    kernel::lu_no_pivot_solve_transpose_multiple(solution.data(), factor.packed.data(),
                                                      factor.size(), solution.cols());
}

} // namespace num
