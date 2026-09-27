/// @file linear/factorization/probed_inverse_diagonal.hpp
/// @brief Randomized estimate of diag(A^-1) for a nonsingular M-matrix.
#pragma once

#include "linear/eigen/lanczos.hpp"
#include "linear/factorization/woodbury.hpp"
#include "linear/graph/randommat/preconditioner.hpp"
#include "linear/solvers/auto_linear.hpp"
#include "linear/sparse/sparse.hpp"
#include "operator/properties.hpp"
#include "stochastic/probe.hpp"
#include <cmath>
#include <optional>
#include <stdexcept>

namespace num {

/// Accuracy and sampling choices for an approximate inverse diagonal.
struct inverse_diagonal_options {
    idx probes = 40;        ///< Gaussian right-hand sides h.
    idx lanczos_steps = 64; ///< Maximum Krylov steps for the matrix square root.
    real tolerance = 1e-8;  ///< Relative tolerance between consecutive Lanczos iterates.
    unsigned seed = 42;     ///< Seed for the probe block and the preconditioner.
};

namespace detail {

/// \f$\tilde{S} = C^{-1} M C^{-T}\f$ for an approximate Cholesky factor C of M,
/// applied without forming either the product or its square root.
class preconditioned_symmetric_operator final {
  public:
    using domain_type = vec;
    using codomain_type = vec;

    preconditioned_symmetric_operator(const spmat &matrix, const grounded_approx_chol_factor &c)
        : matrix_(matrix), factor_(c) {}

    void apply(const vec &input, vec &output) const {
        const vec upper = factor_.solve_upper(input);
        vec product(rows(), 0.0);
        sparse_matvec(matrix_, upper, product);
        output = factor_.solve_lower(product);
    }

    [[nodiscard]] idx rows() const noexcept { return matrix_.n_rows(); }
    [[nodiscard]] idx cols() const noexcept { return matrix_.n_cols(); }

  private:
    const spmat &matrix_;
    const grounded_approx_chol_factor &factor_;
};

} // namespace detail

/// @brief Estimate \f$\operatorname{diag}(A^{-1})\f$ for a nonsingular M-matrix from one block
/// of Gaussian probes, without one solve per entry.
///
/// A diagonal similarity makes the symmetric part positive definite, so each entry is a
/// squared row norm, estimated without bias by the probe mean square. The derivation is in the
/// algorithm notes.
///
/// @param factor A retained factorization of `matrix`.
/// @param matrix The nonsingular M-matrix A, in CSR form.
/// @param symmetrizer \f$\sqrt{\pi}\f$ when A is similar to a symmetric matrix through
///        \f$\operatorname{Diag}(\sqrt{\pi})\f$, which saves the two scaling solves. Empty
///        otherwise.
/// @param options Probe count, Krylov depth, tolerance, and seed.
/// @throws std::invalid_argument If no probes are requested.
/// @throws std::runtime_error If `matrix` is not a nonsingular M-matrix.
template <retained_factorization F>
[[nodiscard]] vec inverse_diagonal(const F &factor, const spmat &matrix,
                                   view<const real> symmetrizer = {},
                                   inverse_diagonal_options options = {}) {
    const idx n = matrix.n_rows();
    if (options.probes == 0) {
        throw std::invalid_argument("inverse_diagonal: at least one probe is required");
    }
    if (matrix.n_cols() != n) {
        throw std::invalid_argument("inverse_diagonal: matrix must be square");
    }
    const bool reversible = symmetrizer.size() != 0;
    if (reversible && symmetrizer.size() != n) {
        throw std::invalid_argument("inverse_diagonal: one weight per row is required");
    }

    // `row_scale` is H with the transformed matrix H A H^-1; `congruence` is W
    // with the grounded Laplacian W S W. Reversibly both are sqrt(pi).
    vec row_scale(n, 0.0), congruence(n, 0.0);
    if (reversible) {
        for (idx j = 0; j < n; ++j) {
            row_scale[j] = symmetrizer[j];
            congruence[j] = symmetrizer[j];
        }
    } else {
        const vec ones(n, 1.0);
        vec q(n, 0.0), r(n, 0.0);
        detail::apply_solve(factor, ones, q);
        detail::apply_solve_transpose(factor, ones, r);
        for (idx j = 0; j < n; ++j) {
            if (!(q[j] > 0.0) || !(r[j] > 0.0)) {
                throw std::runtime_error("inverse_diagonal: matrix is not a nonsingular M-matrix");
            }
            row_scale[j] = std::sqrt(r[j] / q[j]);
            congruence[j] = std::sqrt(r[j] * q[j]);
        }
    }

    // `sparse_diagonal_similarity` forms D^-1 A D, so the weight it takes is the
    // reciprocal of the row scaling H above.
    vec column_scale(n, 0.0);
    for (idx j = 0; j < n; ++j) {
        column_scale[j] = 1.0 / row_scale[j];
    }
    const spmat scaled = sparse_diagonal_similarity(matrix, column_scale);
    const spmat symmetric = reversible ? scaled : symmetric_part(scaled);
    const spmat laplacian = sparse_congruence(symmetric, congruence);

    const grounded_approx_chol_factor approximate = grounded_approxchol_factor(
        laplacian, gao_kyng_spielman_2023::ac2, options.seed ^ 0x9e3779b9U);
    const auto preconditioned =
        num::assume_spd(detail::preconditioned_symmetric_operator(laplacian, approximate));

    // The general case applies A~^-1 to each probe; the reversible case reaches
    // the same operator through the inverse square root and needs no extra factor.
    std::optional<auto_linear_solver> scaled_factor;
    if (!reversible) {
        scaled_factor.emplace(scaled);
    }

    const mat probe = gaussian_probe(n, options.probes, options.seed);
    mat probed(n, options.probes, 0.0);
    for (idx column = 0; column < options.probes; ++column) {
        vec direction(n, 0.0);
        for (idx j = 0; j < n; ++j) {
            direction[j] = probe(j, column);
        }
        vec value(n, 0.0);
        if (reversible) {
            const auto action = inverse_sqrt_lanczos(preconditioned, direction, options.tolerance,
                                                     options.lanczos_steps);
            const vec unscaled = approximate.solve_upper(action.value);
            for (idx j = 0; j < n; ++j) {
                value[j] = row_scale[j] * unscaled[j];
            }
        } else {
            const auto action =
                sqrt_lanczos(preconditioned, direction, options.tolerance, options.lanczos_steps);
            const vec factored = approximate.apply_lower(action.value);
            vec rhs(n, 0.0);
            for (idx j = 0; j < n; ++j) {
                rhs[j] = factored[j] / congruence[j];
            }
            value = solve(*scaled_factor, rhs);
        }
        for (idx j = 0; j < n; ++j) {
            probed(j, column) = value[j];
        }
    }
    return hutchinson_row_mean_square(probed);
}

} // namespace num
