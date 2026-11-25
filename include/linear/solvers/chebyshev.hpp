/// @file linear/solvers/chebyshev.hpp
/// @brief Chebyshev polynomial preconditioner, matrix-free.
#pragma once

#include "container/vector.hpp"
#include "core/math/concepts.hpp"
#include "core/math/laws.hpp"
#include "core/math/operations.hpp"
#include "core/types.hpp"
#include "kernel/kernel.hpp"
#include <cmath>
#include <stdexcept>

namespace num {

/// @brief Largest eigenvalue estimate by power iteration: \f$\lambda_{\max}\f$, times `safety`.
template <class Op>
[[nodiscard]] inline real estimate_largest_eigenvalue(const Op &A, idx iterations = 25,
                                                      real safety = 1.1) {
    const idx n = static_cast<idx>(A.rows());
    if (n == 0) {
        return real(0);
    }
    if (iterations == 0 || !(safety > 0.0)) {
        throw std::invalid_argument("estimate_largest_eigenvalue: invalid iteration or safety");
    }

    vec<real> v(n);
    vec<real> image(n, 0.0);
    // A fixed, non-symmetric starting vector: a constant one is an eigenvector
    // of too many operators of interest (any Laplacian) to be a safe default.
    for (idx i = 0; i < n; ++i) {
        v[i] = real(1) + std::sin(real(i) * real(0.7));
    }

    real lambda = 0.0;
    for (idx step = 0; step < iterations; ++step) {
        const real norm = math::norm(v);
        if (!(norm > 0.0) || !std::isfinite(norm)) {
            return real(0);
        }
        math::scale(real(1) / norm, v);
        math::apply(A, v, image);
        lambda = math::inner(v, image); // Rayleigh quotient, v already unit norm
        v = image;
    }
    return std::abs(lambda) * safety;
}

/// @brief Chebyshev preconditioner: \f$M^{-1} = p_m(A)\f$.
/// \f$p_m\f$ minimizes \f$\max_{\lambda \in [\ell, h]} |1 - \lambda p(\lambda)|\f$.
/// @warning SPD only if \f$[\ell, h]\f$ encloses the spectrum; otherwise PCG may diverge.
/// @warning References `A`; it must outlive the preconditioner.
template <class Op>
requires math::spd_operator<Op>
class chebyshev_preconditioner final {
  public:
    using domain_type = vec<real>;
    using codomain_type = vec<real>;
    /// @brief Declared SPD; valid only when the bounds enclose the spectrum.
    using laws = law::list<law::spd>;

    /// @brief Chebyshev preconditioner of degree `degree` on \f$[\ell, h]\f$.
    /// @throws std::invalid_argument If \f$0 < \ell < h\f$ fails.
    chebyshev_preconditioner(const Op &A, real lo, real hi, idx degree)
        : op_(&A), lo_(lo), hi_(hi), degree_(degree), n_(static_cast<idx>(A.rows())),
          residual_(n_, 0.0), direction_(n_, 0.0), work_(n_, 0.0) {
        if (A.rows() != A.cols()) {
            throw std::invalid_argument("chebyshev: operator must be square");
        }
        if (degree_ == 0) {
            throw std::invalid_argument("chebyshev: degree must be at least 1");
        }
        if (!(lo_ > 0.0) || !(hi_ > lo_) || !std::isfinite(hi_)) {
            throw std::invalid_argument(
                "chebyshev: spectral bounds must satisfy 0 < lo < hi and be finite");
        }
    }

    [[nodiscard]] idx rows() const noexcept { return n_; }
    [[nodiscard]] idx cols() const noexcept { return n_; }
    [[nodiscard]] idx degree() const noexcept { return degree_; }

    /// @brief Preconditioner application: \f$z \leftarrow p_m(A) r\f$.
    void apply(const vec<real> &r, vec<real> &z) const {
        if (r.size() != n_) {
            throw std::invalid_argument("chebyshev: dimension mismatch");
        }
        if (z.size() != n_) {
            z = vec<real>(n_, 0.0);
        }

        const real centre = 0.5 * (hi_ + lo_);
        const real half_width = 0.5 * (hi_ - lo_);
        const real sigma = centre / half_width;
        real rho = real(1) / sigma;

        // Degree 1: the scaled Richardson step z = r / centre.
        z = r;
        math::scale(real(1) / centre, z);
        if (degree_ == 1) {
            return;
        }

        // residual <- r - A z
        math::apply(*op_, z, work_);
        residual_ = r;
        math::axpy(real(-1), work_, residual_);

        // The first direction is the first iterate itself (Saad, Alg. 12.1:
        // d_0 = r/theta), not zero. Starting it at zero drops a term from the
        // recurrence and yields a polynomial that is negative on part of the
        // interval — an indefinite preconditioner rather than an SPD one.
        kernel::copy(direction_.data(), z.data(), n_);

        for (idx k = 1; k < degree_; ++k) {
            const real rho_next = real(1) / ((real(2) * sigma) - rho);
            // direction <- (rho * rho_next) direction + (2 rho_next / half_width) residual
            math::linear_combination(real(2) * rho_next / half_width, residual_, rho * rho_next,
                                     direction_);
            math::axpy(real(1), direction_, z);
            if (k + 1 < degree_) {
                math::apply(*op_, direction_, work_);
                math::axpy(real(-1), work_, residual_);
            }
            rho = rho_next;
        }
    }

  private:
    const Op *op_;
    real lo_;
    real hi_;
    idx degree_;
    idx n_;
    mutable vec<real> residual_;
    mutable vec<real> direction_;
    mutable vec<real> work_;
};

/// @brief Chebyshev preconditioner on \f$[\ell, h]\f$.
template <class Op>
requires math::spd_operator<Op>
[[nodiscard]] inline chebyshev_preconditioner<Op> make_chebyshev_preconditioner(const Op &A, real lo,
                                                                         real hi,
                                                                         idx degree = 4) {
    return chebyshev_preconditioner<Op>(A, lo, hi, degree);
}

/// @brief Chebyshev preconditioner on \f$[\lambda_{\min}, \hat{\lambda}_{\max}]\f$.
/// \f$\lambda_{\max}\f$ is estimated by power iteration.
/// @warning `lambda_min` must be a true lower bound; too large a value makes the
/// polynomial indefinite.
template <class Op>
[[nodiscard]] inline chebyshev_preconditioner<Op>
chebyshev_preconditioner_from_below(const Op &A, real lambda_min, idx degree = 4) {
    const real hi = estimate_largest_eigenvalue(A);
    if (!(hi > 0.0)) {
        throw std::invalid_argument("chebyshev: could not estimate the spectral radius");
    }
    if (!(lambda_min > 0.0) || lambda_min >= hi) {
        throw std::invalid_argument(
            "chebyshev: lambda_min must be positive and below the estimated spectral radius");
    }
    return chebyshev_preconditioner<Op>(A, lambda_min, hi, degree);
}

} // namespace num

