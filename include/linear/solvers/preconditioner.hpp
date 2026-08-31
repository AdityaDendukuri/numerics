/// @file linear/solvers/preconditioner.hpp
/// @brief Preconditioner concept and diagonal preconditioners.
/// @todo Add SSOR, incomplete Cholesky, ILU(0), and block-Jacobi
/// preconditioners for sparse systems.
#pragma once

#include "container/matrix.hpp"
#include "container/vector.hpp"
#include "core/math/evidence.hpp"
#include "core/math/models.hpp"
#include "kernel/raw.hpp"
#include "linear/concepts.hpp"
#include "linear/sparse/sparse.hpp"
#include <cmath>
#include <concepts>
#include <stdexcept>
#include <utility>

namespace num {

/// Diagonal inverse preconditioner.
class JacobiPreconditioner final {
  public:
    using domain_type = Vector;
    using codomain_type = Vector;
    using math_propositions = math::type_list<axiom::positive_definite>;

    /// Take ownership of a precomputed inverse diagonal.
    explicit JacobiPreconditioner(Vector inv_diag) : inv_diag_(std::move(inv_diag)) {
        for (const real value : inv_diag_) {
            if (!(value > 0.0) || !std::isfinite(value)) {
                throw std::invalid_argument(
                    "JacobiPreconditioner: inverse diagonal must be positive and finite");
            }
        }
    }

    [[nodiscard]] idx rows() const noexcept { return inv_diag_.size(); }
    [[nodiscard]] idx cols() const noexcept { return inv_diag_.size(); }

    /// Compute z=D^-1 r.
    void apply(const Vector &r, Vector &z) const {
        const idx n = inv_diag_.size();
        if (r.size() != n) {
            throw std::invalid_argument("JacobiPreconditioner: dimension mismatch");
        }
        if (z.size() != n) {
            z = Vector(n, 0.0);
        }
        kernel::raw::hadamard_mul(z.data(), inv_diag_.data(), r.data(), n);
    }

  private:
    Vector inv_diag_;
};

/// Construct a Jacobi preconditioner from a dense matrix diagonal.
[[nodiscard]] inline JacobiPreconditioner jacobi_preconditioner(const Matrix &A) {
    if (A.rows() != A.cols()) {
        throw std::invalid_argument("jacobi_preconditioner: matrix must be square");
    }
    Vector inv(A.rows());
    for (idx i = 0; i < A.rows(); ++i) {
        if (std::abs(A(i, i)) < real(1e-15)) {
            throw std::invalid_argument("jacobi_preconditioner: zero diagonal");
        }
        inv[i] = real(1) / A(i, i);
    }
    return JacobiPreconditioner(std::move(inv));
}

/// Construct a Jacobi preconditioner from a sparse matrix diagonal.
[[nodiscard]] inline JacobiPreconditioner jacobi_preconditioner(const SparseMatrix &A) {
    if (A.n_rows() != A.n_cols()) {
        throw std::invalid_argument("jacobi_preconditioner: matrix must be square");
    }
    Vector inv(A.n_rows(), 0.0);
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
    return JacobiPreconditioner(std::move(inv));
}

} // namespace num

namespace num::math {

template <>
struct model_of<JacobiPreconditioner> {
    using laws = type_list<law::linear_map>;
};

} // namespace num::math
