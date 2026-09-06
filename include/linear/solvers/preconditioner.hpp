/// @file linear/solvers/preconditioner.hpp
/// @brief preconditioner concept and diagonal preconditioners.
/// @todo Add SSOR, incomplete Cholesky, ILU(0), and block-Jacobi
/// preconditioners for sparse systems.
#pragma once

#include "container/matrix.hpp"
#include "container/vector.hpp"
#include "core/math/evidence.hpp"
#include "core/math/models.hpp"
#include "kernel/kernel.hpp"
#include "linear/concepts.hpp"
#include "linear/sparse/sparse.hpp"
#include <cmath>
#include <concepts>
#include <stdexcept>
#include <utility>

namespace num {

/// Diagonal inverse preconditioner.
class jacobi_preconditioner final {
  public:
    using domain_type = vec;
    using codomain_type = vec;
    using math_laws = math::type_list<law::spd>;

    /// Take ownership of a precomputed inverse diagonal.
    explicit jacobi_preconditioner(vec inv_diag) : inv_diag_(std::move(inv_diag)) {
        for (const real value : inv_diag_) {
            if (!(value > 0.0) || !std::isfinite(value)) {
                throw std::invalid_argument(
                    "jacobi_preconditioner: inverse diagonal must be positive and finite");
            }
        }
    }

    [[nodiscard]] idx rows() const noexcept { return inv_diag_.size(); }
    [[nodiscard]] idx cols() const noexcept { return inv_diag_.size(); }

    /// Compute z=D^-1 r.
    void apply(const vec &r, vec &z) const {
        const idx n = inv_diag_.size();
        if (r.size() != n) {
            throw std::invalid_argument("jacobi_preconditioner: dimension mismatch");
        }
        if (z.size() != n) {
            z = vec(n, 0.0);
        }
        kernel::hadamard_mul(z.data(), inv_diag_.data(), r.data(), n);
    }

  private:
    vec inv_diag_;
};

/// Construct a Jacobi preconditioner from a dense matrix diagonal.
[[nodiscard]] inline jacobi_preconditioner make_jacobi_preconditioner(const mat &A) {
    if (A.rows() != A.cols()) {
        throw std::invalid_argument("jacobi_preconditioner: matrix must be square");
    }
    vec inv(A.rows());
    for (idx i = 0; i < A.rows(); ++i) {
        if (std::abs(A(i, i)) < real(1e-15)) {
            throw std::invalid_argument("jacobi_preconditioner: zero diagonal");
        }
        inv[i] = real(1) / A(i, i);
    }
    return jacobi_preconditioner(std::move(inv));
}

/// Construct a Jacobi preconditioner from a sparse matrix diagonal.
[[nodiscard]] inline jacobi_preconditioner make_jacobi_preconditioner(const spmat &A) {
    if (A.n_rows() != A.n_cols()) {
        throw std::invalid_argument("jacobi_preconditioner: matrix must be square");
    }
    vec inv(A.n_rows(), 0.0);
    for (idx i = 0; i < A.n_rows(); ++i) {
        const idx row_begin = A.row_ptr()[i];
        const idx row_end = A.row_ptr()[i + 1];
        for (idx p = row_begin; p < row_end; ++p) {
            if (A.col_idx()[p] == i) {
                inv[i] = A.values()[p];
                break;
            }
        }
        if (std::abs(inv[i]) < real(1e-15)) {
            throw std::invalid_argument("jacobi_preconditioner: zero diagonal");
        }
        inv[i] = real(1) / inv[i];
    }
    return jacobi_preconditioner(std::move(inv));
}

} // namespace num

namespace num::math {

template <>
struct claims_of<jacobi_preconditioner> {
    using type = type_list<law::linear_map>;
};

} // namespace num::math
