/// @file linear/solvers/chebyshev.hpp
/// @brief mat-free Chebyshev polynomial preconditioner.
///
/// Every other preconditioner here needs the matrix entries: Jacobi reads the
/// diagonal, ApproxChol eliminates a graph, an incomplete factorization walks
/// the sparsity pattern. None of them can precondition an operator built with
/// `num::operators::make_op`, which is the library's own headline pattern — a
/// discrete Laplacian written as a stencil has no entries to read.
///
/// A polynomial preconditioner needs nothing but the operator's action. It
/// approximates \f$A^{-1}\f$ by \f$p_m(A)\f$, where \f$p_m\f$ is the degree-m
/// polynomial minimising \f$\max_{\lambda \in [\ell, h]} |1 - \lambda p(\lambda)|\f$ —
/// the Chebyshev problem, whose solution is available as a three-term
/// recurrence costing one operator application per degree and no setup at all.
///
/// ### The precondition, and what happens when it is wrong
///
/// \f$p_m(A)\f$ is positive definite exactly when \f$p_m > 0\f$ across the
/// spectrum, which holds when \f$0 < \ell \le \lambda_{\min}\f$ and
/// \f$\lambda_{\max} \le h\f$. bounds that exclude part of the spectrum produce
/// an *indefinite* preconditioner, and preconditioned CG built on one silently
/// loses the property that makes its error monotone.
///
/// This is why both bounds are the caller's to supply: they are a mathematical
/// claim, so they go through the same channel as every other one in this
/// library. `chebyshev_preconditioner` asserts positive definiteness, `num::pcg`
/// requires that assertion, and the runtime property sampler rejects the
/// operator if the claim was false. `estimate_largest_eigenvalue` will find
/// \f$h\f$ for you; there is no equally cheap way to find \f$\ell\f$, and
/// guessing it from an assumed condition number is how this preconditioner is
/// most often gotten wrong.
///
/// ### What this is, and is not, good for
///
/// A degree-m polynomial can only improve a condition number by roughly a factor
/// of m. On a second-order elliptic operator, where \f$\kappa\f$ grows with the
/// mesh, that is not a competitive preconditioner and no degree makes it one —
/// use ApproxChol for SDD systems, or an algebraic multigrid for the rest.
///
/// It earns its place in three situations a factorization cannot serve:
///   - **matrix-free operators**, where there are no entries to factor at all;
///   - **moderately conditioned systems** — mass matrices, shifted or damped
///     operators, regularized least squares — where a factor of m is the whole
///     problem;
///   - as a **smoother**, damping the upper spectrum inside a multigrid cycle.
///
/// It also uses no global reductions, which is why polynomial preconditioning
/// keeps reappearing in distributed solvers where those dominate the cost.
#pragma once

#include "container/vector.hpp"
#include "core/math/evidence.hpp"
#include "core/math/models.hpp"
#include "core/math/operations.hpp"
#include "core/types.hpp"
#include "kernel/kernel.hpp"
#include <cmath>
#include <stdexcept>

namespace num {

/// @brief Estimate \f$\lambda_{\max}\f$ of a self-adjoint operator by power iteration.
///
/// Needs only the operator's action, so it works on a matrix-free operator. The
/// starting vector is fixed, so repeated calls on the same operator agree.
///
/// Power iteration approaches the dominant eigenvalue **from below**, so the
/// raw Rayleigh quotient is an under-estimate and unsafe as an upper bound.
/// `safety` inflates the result; the default leaves 10% of headroom.
///
/// @param A Self-adjoint linear operator.
/// @param iterations Power iterations to run.
/// @param safety Multiplier applied to the converged Rayleigh quotient.
/// @return An estimate of \f$\lambda_{\max}\f$, or 0 for an empty operator.
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

    vec v(n);
    vec image(n, 0.0);
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

/// @brief Chebyshev polynomial preconditioner \f$M^{-1} = p_m(A)\f$.
///
/// Applies `degree` operator applications per invocation and allocates nothing
/// after construction. Holds a reference to the operator, which must outlive it.
///
/// @tparam Op Self-adjoint linear operator type.
template <class Op>
class chebyshev_preconditioner final {
  public:
    using domain_type = vec;
    using codomain_type = vec;
    /// Valid only for bounds that enclose the spectrum; see the file comment.
    using math_laws = math::type_list<law::spd>;

    /// @brief Build a degree-`degree` Chebyshev preconditioner for the spectrum \f$[lo, hi]\f$.
    /// @throws std::invalid_argument If the interval is not a positive, non-degenerate range.
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

    /// @brief Apply \f$z \leftarrow p_m(A)\, r\f$.
    void apply(const vec &r, vec &z) const {
        if (r.size() != n_) {
            throw std::invalid_argument("chebyshev: dimension mismatch");
        }
        if (z.size() != n_) {
            z = vec(n_, 0.0);
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
    mutable vec residual_;
    mutable vec direction_;
    mutable vec work_;
};

/// @brief Build a Chebyshev preconditioner over an explicit spectral interval.
///
/// The returned object asserts positive definiteness, which holds only when
/// \f$[lo, hi]\f$ encloses the spectrum. See the file comment.
template <class Op>
[[nodiscard]] inline chebyshev_preconditioner<Op> make_chebyshev_preconditioner(const Op &A, real lo,
                                                                         real hi,
                                                                         idx degree = 4) {
    return chebyshev_preconditioner<Op>(A, lo, hi, degree);
}

/// @brief Build a Chebyshev preconditioner from a known \f$\lambda_{\min}\f$,
/// estimating only the upper bound.
///
/// Power iteration gives \f$\lambda_{\max}\f$ cheaply and reliably. There is no
/// equally cheap estimate of \f$\lambda_{\min}\f$, and it is deliberately *not*
/// inferred here from an assumed condition number: too large a value excludes
/// the bottom of the spectrum, which makes the polynomial indefinite and
/// destroys the property PCG depends on. Supply a genuine lower bound — from the
/// problem's physics, from a shift or regularization parameter, or from
/// `num::lanczos` — or use the two-bound overload directly.
///
/// @param A Self-adjoint linear operator.
/// @param lambda_min A true lower bound on the spectrum, greater than zero.
/// @param degree Polynomial degree, and operator applications per solve.
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

namespace num::math {

template <class Op>
struct claims_of<chebyshev_preconditioner<Op>> {
    using type = type_list<law::linear_map>;
};

namespace detail {

template <class Op>
struct domain_of<chebyshev_preconditioner<Op>, void> {
    using type = vec;
};

template <class Op>
struct codomain_of<chebyshev_preconditioner<Op>, void> {
    using type = vec;
};

} // namespace detail

} // namespace num::math
