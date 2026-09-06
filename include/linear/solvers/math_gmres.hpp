/// @file math_gmres.hpp
/// @brief Generic restarted GMRES constrained by the linear-map model.
#pragma once

#include "core/types.hpp"
#include "core/math/concepts.hpp"
#include "linear/solvers/solver_result.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <concepts>
#include <stdexcept>
#include <vector>

namespace num {

struct gmres_options {
    real tolerance = 1e-6;
    idx max_iterations = 1000;
    idx restart = 30;
};

/// Restarted GMRES over a real inner-product space and certified linear map.
template <class Op, class V>
requires math::inner_product_space<V> &&math::endomorphism_on<Op, V> &&
    std::same_as<math::scalar_t<V>, real> [[nodiscard]] solver_result
    gmres(const Op &A, const V &b, V &x, gmres_options options = {}) {
    const auto dimension = math::dimension(b);
    if (math::dimension(x) != dimension || A.rows() != dimension || A.cols() != dimension) {
        throw std::invalid_argument("gmres: incompatible operator and vector dimensions");
    }
    if (!(options.tolerance > 0.0) || options.restart == 0) {
        throw std::invalid_argument("gmres: invalid convergence options");
    }
    const idx n = static_cast<idx>(dimension);
    const idx restart = std::min(options.restart, n);
    solver_result result{0, 0.0, false};
    V residual = math::zero_like(b);
    V scratch = math::zero_like(b);
    array<V> basis;
    basis.reserve(restart + 1);
    array<array<real>> hessenberg(restart, array<real>(restart + 1, 0.0));
    array<real> cosine(restart, 0.0), sine(restart, 0.0), rhs(restart + 1, 0.0);
    idx total_iterations = 0;

    while (true) {
        math::apply(A, x, residual);
        // r <- b - A*x
        math::linear_combination(real(1), b, real(-1), residual);
        const real initial_norm = math::norm(residual);
        result.residual = initial_norm;
        if (initial_norm < options.tolerance) {
            result.converged = true;
            break;
        }
        if (total_iterations >= options.max_iterations)
            break;

        basis.clear();
        basis.push_back(residual);
        math::scale(real(1) / initial_norm, basis[0]);
        for (auto &column : hessenberg)
            std::fill(column.begin(), column.end(), 0.0);
        std::fill(cosine.begin(), cosine.end(), 0.0);
        std::fill(sine.begin(), sine.end(), 0.0);
        std::fill(rhs.begin(), rhs.end(), 0.0);
        rhs[0] = initial_norm;

        idx steps = 0;
        for (; steps < restart && total_iterations < options.max_iterations; ++steps) {
            const idx j = steps;
            ++total_iterations;
            result.iterations = total_iterations;
            math::apply(A, basis[j], scratch);
            for (idx i = 0; i <= j; ++i) {
                // h_{i,j} <- v_i^T*w
                hessenberg[j][i] = math::inner(basis[i], scratch);
                // w <- w - h_{i,j}*v_i
                math::axpy(-hessenberg[j][i], basis[i], scratch);
            }
            hessenberg[j][j + 1] = math::norm(scratch);

            for (idx i = 0; i < j; ++i) {
                const real upper = hessenberg[j][i];
                const real lower = hessenberg[j][i + 1];
                hessenberg[j][i] = cosine[i] * upper + sine[i] * lower;
                hessenberg[j][i + 1] = -sine[i] * upper + cosine[i] * lower;
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

            const bool arnoldi_breakdown = subdiagonal <= real(1e-15);
            if (!arnoldi_breakdown) {
                math::scale(real(1) / subdiagonal, scratch);
                basis.push_back(scratch);
            }
            if (result.residual < options.tolerance)
                result.converged = true;
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
            if (!(std::abs(pivot) > real(1e-15)) || !std::isfinite(pivot)) {
                throw std::runtime_error("gmres: projected triangular system is singular");
            }
            coefficients[row] /= pivot;
        }
        for (idx i = 0; i < steps; ++i)
            math::axpy(coefficients[i], basis[i], x);
        if (result.converged)
            break;
    }
    return result;
}

/// @brief Right-preconditioned restarted GMRES: solves \f$A M^{-1} u = b\f$, \f$x = M^{-1} u\f$.
///
/// Right preconditioning rather than left, because it leaves the residual alone.
/// Under left preconditioning the Arnoldi process minimises \f$\|M^{-1}(b-Ax)\|\f$,
/// so the quantity the stopping test sees is not the residual the caller asked
/// about, and a badly scaled `M` makes the solver stop early or late for reasons
/// that have nothing to do with the problem. Here `result.residual` remains
/// \f$\|b - Ax\|_2\f$ throughout.
///
/// The preconditioner carries no symmetry requirement — an ILU(0) is not
/// self-adjoint even for symmetric `A` — so this takes any linear operator of
/// matching dimension.
///
/// @tparam Op Linear operator type.
/// @tparam M preconditioner type; only its action is used.
/// @tparam V vec space type.
/// @param A System operator.
/// @param preconditioner Approximate inverse applied on the right.
/// @param b Right-hand side.
/// @param x Solution; initial guess on input.
/// @param options Tolerance, restart length, and iteration limit.
template <class Op, class M, class V>
requires math::inner_product_space<V> &&math::endomorphism_on<Op, V> &&math::endomorphism_on<M, V> &&
    std::floating_point<math::scalar_t<V>> [[nodiscard]] solver_result
    gmres(const Op &A, const M &preconditioner, const V &b, V &x, gmres_options options = {}) {
    const auto dimension = math::dimension(b);
    if (math::dimension(x) != dimension || A.rows() != dimension || A.cols() != dimension ||
        preconditioner.rows() != dimension || preconditioner.cols() != dimension) {
        throw std::invalid_argument(
            "gmres: incompatible operator, preconditioner, and vector dimensions");
    }
    if (!(options.tolerance > 0.0) || options.restart == 0) {
        throw std::invalid_argument("gmres: invalid convergence options");
    }
    const idx n = static_cast<idx>(dimension);
    const idx restart = std::min(options.restart, n);
    solver_result result{0, 0.0, false};

    V residual = math::zero_like(b);
    V scratch = math::zero_like(b);
    V preconditioned = math::zero_like(b);
    array<V> basis;      // Arnoldi basis of the Krylov space of A M^-1
    array<V> images;     // M^-1 v_j, kept so the update needs no second solve
    basis.reserve(restart + 1);
    images.reserve(restart);
    array<array<real>> hessenberg(restart, array<real>(restart + 1, 0.0));
    array<real> cosine(restart, 0.0), sine(restart, 0.0), rhs(restart + 1, 0.0);
    idx total_iterations = 0;

    while (true) {
        math::apply(A, x, residual);
        math::linear_combination(real(1), b, real(-1), residual);
        const real initial_norm = math::norm(residual);
        result.residual = initial_norm;
        if (initial_norm < options.tolerance) {
            result.converged = true;
            break;
        }
        if (total_iterations >= options.max_iterations) {
            break;
        }

        basis.clear();
        images.clear();
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
            math::apply(preconditioner, basis[j], preconditioned);
            images.push_back(preconditioned);
            math::apply(A, images[j], scratch);

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

            // Relative to the residual that started this cycle: an absolute floor
            // would call breakdown on every step of a small-scaled problem.
            const bool arnoldi_breakdown =
                subdiagonal <= std::numeric_limits<real>::epsilon() * initial_norm;
            if (!arnoldi_breakdown) {
                math::scale(real(1) / subdiagonal, scratch);
                basis.push_back(scratch);
            }
            if (result.residual < options.tolerance) {
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
            if (!(std::abs(pivot) > real(1e-15)) || !std::isfinite(pivot)) {
                throw std::runtime_error("gmres: projected triangular system is singular");
            }
            coefficients[row] /= pivot;
        }
        // x += sum_i y_i M^-1 v_i -- the preconditioned basis, not the Arnoldi one.
        for (idx i = 0; i < steps; ++i) {
            math::axpy(coefficients[i], images[i], x);
        }
        if (result.converged) {
            break;
        }
    }
    return result;
}

} // namespace num
