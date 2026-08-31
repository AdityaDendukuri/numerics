/// @file math_gmres.hpp
/// @brief Generic restarted GMRES constrained by the linear-map model.
#pragma once

#include "core/math/concepts.hpp"
#include "linear/solvers/solver_result.hpp"
#include <algorithm>
#include <cmath>
#include <concepts>
#include <stdexcept>
#include <vector>

namespace num {

struct GMRESOptions {
    real tolerance = 1e-6;
    idx max_iterations = 1000;
    idx restart = 30;
};

/// Restarted GMRES over a real inner-product space and certified linear map.
template <class Op, class V>
requires math::InnerProductSpace<V> &&math::EndomorphismOn<Op, V> &&
    std::same_as<math::scalar_t<V>, real> [[nodiscard]] SolverResult
    gmres(const Op &A, const V &b, V &x, GMRESOptions options = {}) {
    const auto dimension = math::dimension(b);
    if (math::dimension(x) != dimension || A.rows() != dimension || A.cols() != dimension) {
        throw std::invalid_argument("gmres: incompatible operator and vector dimensions");
    }
    if (!(options.tolerance > 0.0) || options.restart == 0) {
        throw std::invalid_argument("gmres: invalid convergence options");
    }
    const idx n = static_cast<idx>(dimension);
    const idx restart = std::min(options.restart, n);
    SolverResult result{0, 0.0, false};
    V residual = math::zero_like(b);
    V scratch = math::zero_like(b);
    std::vector<V> basis;
    basis.reserve(restart + 1);
    std::vector<std::vector<real>> hessenberg(restart, std::vector<real>(restart + 1, 0.0));
    std::vector<real> cosine(restart, 0.0), sine(restart, 0.0), rhs(restart + 1, 0.0);
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

        std::vector<real> coefficients(steps, 0.0);
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

} // namespace num
