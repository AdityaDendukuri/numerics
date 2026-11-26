/// @file linear/factorization/probed_inverse_diagonal.hpp
/// @brief Randomized inverse diagonal of an M-matrix: \f$\mathrm{diag}(A^{-1})\f$.
#pragma once

#include "linear/eigen/lanczos.hpp"
#include "linear/factorization/woodbury.hpp"
#include "linear/graph/randommat/preconditioner.hpp"
#include "linear/factorization/lu.hpp"
#include "linear/sparse/sparse.hpp"
#include "operator/properties.hpp"
#include "stochastic/probe.hpp"
#include <cmath>
#include <stdexcept>

namespace num {

/// @brief Sampler for the approximate Cholesky factor: Gao–Kyng–Spielman AC or AC2.
enum class approximate_cholesky_preconditioner { ac, ac2 };
/// @brief Use the AC sampler: one sample per edge, cheaper.
inline constexpr approximate_cholesky_preconditioner ac = approximate_cholesky_preconditioner::ac;
/// @brief Use the AC2 sampler: two samples per edge, more accurate. The default.
inline constexpr approximate_cholesky_preconditioner ac2 = approximate_cholesky_preconditioner::ac2;

/// @brief Options for estimating the diagonal of the inverse: probes, Lanczos steps, tolerance,
/// seed.
struct inverse_diagonal_options {
    approximate_cholesky_preconditioner preconditioner = ac2; ///< AC or AC2.
    idx probes = 40;                                          ///< Gaussian right-hand sides h.
    idx lanczos_steps = 64; ///< Maximum Krylov steps for the matrix square root.
    real tolerance = 1e-8;  ///< Relative tolerance between consecutive Lanczos iterates.
    unsigned seed = 42;     ///< Seed for the probe block and the preconditioner.
};

/// @brief Compute an approximate Cholesky factor with a sampler chosen at run time:
/// \f$A \approx C C^T\f$.
[[nodiscard]] inline grounded_approx_chol_factor
grounded_approxchol_factor(const spmat &matrix, approximate_cholesky_preconditioner method,
                           unsigned seed) {
    if (method == ac) {
        return grounded_approxchol_factor(matrix, gao_kyng_spielman_2023::ac, seed);
    }
    return grounded_approxchol_factor(matrix, gao_kyng_spielman_2023::ac2, seed);
}

namespace detail {

/// @brief Operator \f$\tilde{S} = C^{-1} M C^{-T}\f$, applied without forming it.
class preconditioned_symmetric_operator final {
  public:
    using domain_type = vec<real>;
    using codomain_type = vec<real>;

    preconditioned_symmetric_operator(const spmat &matrix, const grounded_approx_chol_factor &c)
        : matrix_(matrix), factor_(c) {}

    void apply(const vec<real> &input, vec<real> &output) const {
        const vec<real> upper = factor_.solve_upper(input);
        vec<real> product(rows(), 0.0);
        sparse_matvec(matrix_, upper, product);
        output = factor_.solve_lower(product);
    }

    [[nodiscard]] idx rows() const noexcept { return matrix_.n_rows(); }
    [[nodiscard]] idx cols() const noexcept { return matrix_.n_cols(); }

  private:
    const spmat &matrix_;
    const grounded_approx_chol_factor &factor_;
};

inline vec<real> reversible_inverse_diagonal(const spmat &matrix, view<const real> symmetrizer,
                                             const grounded_approx_chol_factor &approximate,
                                             inverse_diagonal_options options) {
    const idx n = matrix.n_rows();
    vec<real> inverse_scale(n, 0.0);
    for (idx j = 0; j < n; ++j) {
        inverse_scale[j] = 1.0 / symmetrizer[j];
    }
    const spmat symmetric = sparse_diagonal_similarity(matrix, inverse_scale);
    const spmat laplacian = sparse_congruence(symmetric, symmetrizer);
    const auto preconditioned =
        num::assume_spd(preconditioned_symmetric_operator(laplacian, approximate));

    const mat<real> probe = gaussian_probe(n, options.probes, options.seed);
    mat<real> probed(n, options.probes, 0.0);
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if (options.probes > 1)
#endif
    for (idx column = 0; column < options.probes; ++column) {
        vec<real> direction(n, 0.0);
        for (idx j = 0; j < n; ++j) {
            direction[j] = probe(j, column);
        }
        const auto action = inverse_sqrt_lanczos(preconditioned, direction, options.tolerance,
                                                 options.lanczos_steps);
        const vec<real> unscaled = approximate.solve_upper(action.value);
        for (idx j = 0; j < n; ++j) {
            probed(j, column) = symmetrizer[j] * unscaled[j];
        }
    }
    return hutchinson_row_mean_square(probed);
}

} // namespace detail

/// @brief Estimate the diagonal of the inverse of an M-matrix from random probes and Lanczos:
/// \f$\mathrm{diag}(A^{-1})\f$.
[[nodiscard]] inline vec<real> inverse_diagonal(const spmat &matrix, view<const real> symmetrizer,
                                                inverse_diagonal_options options = {}) {
    const idx n = matrix.n_rows();
    if (options.probes == 0) {
        throw std::invalid_argument("inverse_diagonal: at least one probe is required");
    }
    if (matrix.n_cols() != n) {
        throw std::invalid_argument("inverse_diagonal: matrix must be square");
    }
    if (symmetrizer.size() != n) {
        throw std::invalid_argument("inverse_diagonal: one weight per row is required");
    }
    vec<real> inverse_scale(n, 0.0);
    for (idx j = 0; j < n; ++j) {
        inverse_scale[j] = 1.0 / symmetrizer[j];
    }
    const spmat symmetric = sparse_diagonal_similarity(matrix, inverse_scale);
    const spmat laplacian = sparse_congruence(symmetric, symmetrizer);
    const grounded_approx_chol_factor approximate =
        grounded_approxchol_factor(laplacian, options.preconditioner, options.seed ^ 0x9e3779b9U);
    return detail::reversible_inverse_diagonal(matrix, symmetrizer, approximate, options);
}

/// @brief Estimate the diagonal of the inverse of a reversible M-matrix, reusing an approximate
/// factor.
[[nodiscard]] inline vec<real> inverse_diagonal(const spmat &matrix, view<const real> symmetrizer,
                                                const grounded_approx_chol_factor &approximate,
                                                inverse_diagonal_options options = {}) {
    if (matrix.n_cols() != matrix.n_rows() || symmetrizer.size() != matrix.n_rows() ||
        approximate.rows() != matrix.n_rows()) {
        throw std::invalid_argument("inverse_diagonal: incompatible matrix, weights, and factor");
    }
    return detail::reversible_inverse_diagonal(matrix, symmetrizer, approximate, options);
}

/// @brief Estimate the diagonal of the inverse from random probes:
/// \f$(A^{-1})_{jj} \approx \frac{1}{p} \sum_k (R z_k)_j^2\f$.
/// `symmetrizer` is \f$\sqrt{\pi}\f$ when \f$A\f$ is symmetrizable by it, else empty.
/// @throws std::invalid_argument If no probes are requested.
/// @throws std::runtime_error If `matrix` is not a nonsingular M-matrix.
template <factorization F>
[[nodiscard]] vec<real> inverse_diagonal(const F &factor, const spmat &matrix,
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
    if (reversible) {
        return inverse_diagonal(matrix, symmetrizer, options);
    }

    // `row_scale` is H with the transformed matrix H A H^-1; `congruence` is W
    // with the grounded Laplacian W S W.
    vec<real> row_scale(n, 0.0), congruence(n, 0.0);
    const vec<real> ones(n, 1.0);
    vec<real> q(n, 0.0), r(n, 0.0);
    solve(factor, ones, q);
    solve_transpose(factor, ones, r);
    for (idx j = 0; j < n; ++j) {
        if (!(q[j] > 0.0) || !(r[j] > 0.0)) {
            throw std::runtime_error("inverse_diagonal: matrix is not a nonsingular M-matrix");
        }
        row_scale[j] = std::sqrt(r[j] / q[j]);
        congruence[j] = std::sqrt(r[j] * q[j]);
    }

    // `sparse_diagonal_similarity` forms D^-1 A D, so the weight it takes is the
    // reciprocal of the row scaling H above.
    vec<real> column_scale(n, 0.0);
    for (idx j = 0; j < n; ++j) {
        column_scale[j] = 1.0 / row_scale[j];
    }
    const spmat scaled = sparse_diagonal_similarity(matrix, column_scale);
    const spmat symmetric = symmetric_part(scaled);
    const spmat laplacian = sparse_congruence(symmetric, congruence);

    const grounded_approx_chol_factor approximate =
        grounded_approxchol_factor(laplacian, options.preconditioner, options.seed ^ 0x9e3779b9U);
    const auto preconditioned =
        num::assume_spd(detail::preconditioned_symmetric_operator(laplacian, approximate));

    const auto scaled_factor = lu(dense(scaled));

    const mat<real> probe = gaussian_probe(n, options.probes, options.seed);
    mat<real> rhs(n, options.probes, 0.0);
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if (options.probes > 1)
#endif
    for (idx column = 0; column < options.probes; ++column) {
        vec<real> direction(n, 0.0);
        for (idx j = 0; j < n; ++j) {
            direction[j] = probe(j, column);
        }
        const auto action =
            sqrt_lanczos(preconditioned, direction, options.tolerance, options.lanczos_steps);
        const vec<real> factored = approximate.apply_lower(action.value);
        for (idx j = 0; j < n; ++j) {
            rhs(j, column) = factored[j] / congruence[j];
        }
    }
    // One block solve for all probes.
    mat<real> probed(n, options.probes, 0.0);
    solve(scaled_factor, rhs, probed);
    return hutchinson_row_mean_square(probed);
}

} // namespace num
