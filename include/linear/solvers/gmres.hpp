/// @file linear/solvers/gmres.hpp
/// @brief Restarted GMRES for any linear operator.
#pragma once

#include "core/math/concepts.hpp"
#include "core/policy.hpp"
#include "core/types.hpp"
#include "linear/math_adapters.hpp"
#include "linear/solvers/solver_result.hpp"
#include "linear/sparse/sparse_op.hpp"
#include "operator/dense.hpp"
#include <algorithm>
#include <cmath>
#include <concepts>
#include <limits>
#include <stdexcept>
#include <vector>

namespace num {

/// @brief Convergence options of `num::gmres`: tolerance, iteration limit, and restart length.
struct gmres_options {
    num::tolerance tolerance{.rtol = 1e-6}; ///< On \f$\|b - Ax\|_2\f$ against \f$\|b\|_2\f$.
    idx max_iterations = 1000;
    idx restart = 30;
};

/// @brief The preconditioner \f$M = I\f$: `gmres(A, no_preconditioner{}, b, x)` is plain GMRES,
/// with the preconditioner's applies and stored images compiled away.
struct no_preconditioner {};

/// @brief Right-preconditioned restarted GMRES: solves \f$A M^{-1} u = b\f$, \f$x = M^{-1} u\f$.
///
/// Right preconditioning keeps `result.residual` equal to \f$\|b - Ax\|_2\f$, so the stopping
/// test does not depend on the scaling of `M`. `M` need not be symmetric, as with ILU(0).
///
/// @param A System operator.
/// @param preconditioner Approximate inverse applied on the right.
/// @param b Right-hand side.
/// @param x Solution; initial guess on input.
/// @param options Tolerance, restart length, and iteration limit.
template <class Op, class M, class V>
requires math::inner_product_space<V> && math::linear_operator<Op, V, V> &&
    (math::linear_operator<M, V, V> || std::same_as<M, no_preconditioner>) &&
    std::floating_point<math::scalar_t<V>> [[nodiscard]] solver_result
    gmres(const Op &A, const M &preconditioner, const V &b, V &x, gmres_options options = {}) {
    constexpr bool preconditioned = !std::same_as<M, no_preconditioner>;
    const auto dimension = math::dimension(b);
    if (math::dimension(x) != dimension || A.rows() != dimension || A.cols() != dimension) {
        throw std::invalid_argument("gmres: incompatible operator and vector dimensions");
    }
    if constexpr (preconditioned) {
        if (preconditioner.rows() != dimension || preconditioner.cols() != dimension) {
            throw std::invalid_argument("gmres: incompatible preconditioner dimensions");
        }
    }
    if (!options.tolerance.valid() || options.restart == 0) {
        throw std::invalid_argument("gmres: invalid convergence options");
    }
    const real b_norm = static_cast<real>(math::norm(b));
    const idx n = static_cast<idx>(dimension);
    const idx restart = std::min(options.restart, n);
    solver_result result{0, 0.0, false};

    V residual = math::zero_like(b);
    V scratch = math::zero_like(b);
    array<V> basis;  // Arnoldi basis of the Krylov space of A M^-1
    array<V> images; // M^-1 v_j, kept so the update needs no second solve; unused when M = I
    basis.reserve(restart + 1);
    if constexpr (preconditioned) {
        images.reserve(restart);
    }
    array<array<real>> hessenberg(restart, array<real>(restart + 1, 0.0));
    array<real> cosine(restart, 0.0), sine(restart, 0.0), rhs(restart + 1, 0.0);
    idx total_iterations = 0;

    while (true) {
        math::apply(A, x, residual);
        math::linear_combination(real(1), b, real(-1), residual);
        const real initial_norm = math::norm(residual);
        result.residual = initial_norm;
        if (options.tolerance.met(initial_norm, b_norm)) {
            result.converged = true;
            break;
        }
        if (total_iterations >= options.max_iterations) {
            break;
        }

        basis.clear();
        if constexpr (preconditioned) {
            images.clear();
        }
        basis.push_back(residual);
        math::scale(real(1) / initial_norm, basis[0]);
        for (auto &column : hessenberg) {
            std::fill(column.begin(), column.end(), 0.0);
        }
        std::fill(cosine.begin(), cosine.end(), 0.0);
        std::fill(sine.begin(), sine.end(), 0.0);
        std::fill(rhs.begin(), rhs.end(), 0.0);
        rhs[0] = initial_norm;

        idx steps = 0;
        for (; steps < restart && total_iterations < options.max_iterations; ++steps) {
            const idx j = steps;
            ++total_iterations;
            result.iterations = total_iterations;

            // w = A M^-1 v_j, with M^-1 v_j retained for the update below.
            if constexpr (preconditioned) {
                images.push_back(math::zero_like(b));
                math::apply(preconditioner, basis[j], images[j]);
                math::apply(A, images[j], scratch);
            } else {
                math::apply(A, basis[j], scratch);
            }
            const real image_norm = math::norm(scratch);

            for (idx i = 0; i <= j; ++i) {
                hessenberg[j][i] = math::inner(basis[i], scratch);
                math::axpy(-hessenberg[j][i], basis[i], scratch);
            }
            hessenberg[j][j + 1] = math::norm(scratch);

            for (idx i = 0; i < j; ++i) {
                const real upper = hessenberg[j][i];
                const real lower = hessenberg[j][i + 1];
                hessenberg[j][i] = (cosine[i] * upper) + (sine[i] * lower);
                hessenberg[j][i + 1] = (-sine[i] * upper) + (cosine[i] * lower);
            }

            const real diagonal = hessenberg[j][j];
            const real subdiagonal = hessenberg[j][j + 1];
            const real radius = std::hypot(diagonal, subdiagonal);
            if (!(radius > 0.0) || !std::isfinite(radius)) {
                throw std::runtime_error("gmres: Arnoldi breakdown produced a singular projection");
            }
            cosine[j] = diagonal / radius;
            sine[j] = subdiagonal / radius;
            hessenberg[j][j] = radius;
            hessenberg[j][j + 1] = 0.0;
            rhs[j + 1] = -sine[j] * rhs[j];
            rhs[j] = cosine[j] * rhs[j];
            result.residual = std::abs(rhs[j + 1]);

            // Breakdown when orthogonalization left only rounding noise of A v_j.
            const bool arnoldi_breakdown = subdiagonal <= num::eps * image_norm;
            if (!arnoldi_breakdown) {
                math::scale(real(1) / subdiagonal, scratch);
                basis.push_back(scratch);
            }
            if (options.tolerance.met(result.residual, b_norm)) {
                result.converged = true;
            }
            if (result.converged || arnoldi_breakdown) {
                ++steps;
                break;
            }
        }

        array<real> coefficients(steps, 0.0);
        for (idx row = steps; row > 0;) {
            --row;
            coefficients[row] = rhs[row];
            for (idx column = row + 1; column < steps; ++column) {
                coefficients[row] -= hessenberg[column][row] * coefficients[column];
            }
            const real pivot = hessenberg[row][row];
            if (!(std::abs(pivot) > real(0)) || !std::isfinite(pivot)) {
                throw std::runtime_error("gmres: projected triangular system is singular");
            }
            coefficients[row] /= pivot;
        }
        // x += sum_i y_i M^-1 v_i -- the preconditioned basis, not the Arnoldi one.
        for (idx i = 0; i < steps; ++i) {
            if constexpr (preconditioned) {
                math::axpy(coefficients[i], images[i], x);
            } else {
                math::axpy(coefficients[i], basis[i], x);
            }
        }
        if (result.converged) {
            break;
        }
    }
    return result;
}

/// @brief Solve general unsymmetric linear systems \f$A x = b\f$ using restarted GMRES.
///
/// Constructs an orthogonal Arnoldi basis \f$V_m\f$ and solves the projected least-squares
/// problem \f$\min \| \beta e_1 - \bar{H}_m y \|_2\f$ with Givens plane rotations.
///
/// @param A General square linear operator (dense, sparse, or matrix-free).
/// @param b Right-hand side vector.
/// @param x Solution vector (serves as initial guess on input, updated in place).
/// @param options Tolerance on \f$\|b - A x\|_2\f$, iteration limit, and restart length.
/// @return `solver_result` containing iteration count, final residual norm, and convergence boolean.
/// @throws std::invalid_argument If dimensions do not match.
/// @see cg, minres, pcg
template <class Op, class V>
requires math::inner_product_space<V> && math::linear_operator<Op, V, V> &&
    std::same_as<math::scalar_t<V>, real> [[nodiscard]] solver_result
    gmres(const Op &A, const V &b, V &x, gmres_options options = {}) {
    return gmres(A, no_preconditioner{}, b, x, options);
}

/// @brief Solve sparse linear systems \f$A x = b\f$ using restarted GMRES.
///
/// @param A Square CSR sparse matrix.
/// @param b Right-hand side vector.
/// @param x Solution vector (serves as initial guess on input, updated in place).
/// @param options Tolerance, iteration limit, and restart length.
/// @return `solver_result` containing iteration count, final residual norm, and convergence boolean.
/// @throws std::invalid_argument If `A` is not square or dimensions mismatch.
inline solver_result gmres(const spmat &A, const vec<real> &b, vec<real> &x, gmres_options options = {}) {
    if (A.n_rows() != A.n_cols()) {
        throw std::invalid_argument("gmres: sparse matrix must be square");
    }
    return gmres(
        operators::sparse_op(A), b, x,
        options);
}

/// @brief Solve dense linear systems \f$A x = b\f$ using restarted GMRES with hardware backend dispatch.
///
/// @param A Square dense matrix.
/// @param b Right-hand side vector.
/// @param x Solution vector (serves as initial guess on input, updated in place).
/// @param options Tolerance, iteration limit, and restart length.
/// @return `solver_result` containing iteration count, final residual norm, and convergence boolean.
/// @throws std::invalid_argument If `A` is not square or dimensions mismatch.
inline solver_result gmres(const mat<real> &A, const vec<real> &b, vec<real> &x, gmres_options options = {}) {
    if (A.rows() != A.cols()) {
        throw std::invalid_argument("gmres: dense matrix must be square");
    }
    return gmres(
        operators::dense_op(A), b, x,
        options);
}

} // namespace num
