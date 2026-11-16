/// @file expv.hpp
/// @brief The action exp(tA)v of the matrix exponential, from a Krylov subspace.
///
/// Approximates \f$\exp(tA)v \approx \|v\| Q_m \exp(tH_m)e_1\f$ where
/// \f$AQ_m \approx Q_{m+1}\bar{H}_m\f$ is the Arnoldi relation.
/// @todo Add adaptive step subdivision and an a posteriori error estimate for
/// large \f$|t|\|A\|\f$.
#pragma once

#include "container/matrix_ops.hpp"
#include "linear/factorization/lu.hpp"
#include "linear/sparse/sparse_op.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <string>

#include "container/vector_ops.hpp"

#include "container/matrix.hpp"
#include "core/types.hpp"
#include "container/vector.hpp"
#include "linear/subspace.hpp"
#include "linear/sparse/sparse.hpp"
#include "operator/concepts.hpp"
#include <stdexcept>
#include <utility>
#include <vector>

namespace num {

namespace detail {
mat<real> dense_expm_pade6(const mat<real> &A);
}

/// @brief Compute \f$\exp(tA)v\f$ for any \f$y=A x\f$ adapter.
template <class Op>
requires linear_operator<Op, vec<real>, vec<real>> vec<real> expv(real t, const Op &A, const vec<real> &v,
                                                        int m_max = 30, real tol = 1e-8) {
    const idx n = A.rows();
    if (A.cols() != n || v.size() != n) {
        throw std::invalid_argument("expv: dimension mismatch");
    }

    real beta = norm(v);
    if (beta == 0.0) {
        return vec<real>(n, 0.0);
    }

    array<vec<real>> V;
    V.reserve(m_max + 1);

    vec<real> v0(n);
    for (idx i = 0; i < n; i++) {
        v0[i] = v[i] / beta;
    }
    V.push_back(std::move(v0));

    mat<real> H(m_max + 1, m_max, 0.0);
    int m_actual = m_max;
    array<real> h_col(m_max + 1, 0.0);

    for (int j = 0; j < m_max; j++) {
        vec<real> w(n, 0.0);
        A.apply(V[j], w);
        const real image_norm = norm(w);

        const real h_next = dispatch::subspace::mgs_orthogonalize(V, w, h_col, j + 1);
        for (int i = 0; i <= j; i++) {
            H(i, j) = h_col[i];
        }
        H(j + 1, j) = h_next;

        if (h_next <= tol * image_norm) { // happy breakdown, relative to ||A v_j||
            m_actual = j + 1;
            break;
        }

        scale(w, real(1) / h_next);
        V.push_back(std::move(w));
    }

    mat<real> projected(m_actual, m_actual, 0.0);
    for (int i = 0; i < m_actual; i++) {
        for (int j = 0; j < m_actual; j++) {
            projected(i, j) = t * H(i, j);
        }
    }

    mat<real> E = detail::dense_expm_pade6(projected);

    vec<real> result(n, 0.0);
    for (int j = 0; j < m_actual; j++) {
        axpy(beta * E(j, 0), V[j], result);
    }

    return result;
}

vec<real> expv(real t, const spmat &A, const vec<real> &v, int m_max = 30, real tol = 1e-8);

namespace detail {

inline mat<real> dense_expm_pade6(const mat<real> &A) {
    const idx m = A.rows();

    static constexpr double c[7] = {1.0,         0.5,           5.0 / 44.0,    1.0 / 66.0,
                                    1.0 / 792.0, 1.0 / 15840.0, 1.0 / 665280.0};

    double norm_inf = 0.0;
    for (idx i = 0; i < m; i++) {
        double row_sum = 0.0;
        for (idx j = 0; j < m; j++) {
            row_sum += std::abs(A(i, j));
        }
        norm_inf = num::max(norm_inf, row_sum);
    }

    int s = 0;
    if (norm_inf > 0.5) {
        s = (int)std::max(0.0, std::ceil(std::log2(norm_inf / 0.5)));
    }

    double scale = std::ldexp(1.0, -s);
    mat<real> as(m, m, 0.0);
    // A_s <- 2^{-s} A.
    kernel::scale_copy_strided(as.data(), 1, A.data(), 1, scale, m * m);

    mat<real> B(m, m, 0.0);
    matmul(as, as, B);

    mat<real> B2(m, m, 0.0);
    matmul(B, B, B2);

    mat<real> B3(m, m, 0.0);
    matmul(B2, B, B3);

    mat<real> v_mat(m, m, 0.0);
    for (idx i = 0; i < m; i++) {
        for (idx j = 0; j < m; j++) {
            v_mat(i, j) = (c[6] * B3(i, j)) + (c[4] * B2(i, j)) + (c[2] * B(i, j));
        }
        v_mat(i, i) += c[0];
    }

    mat<real> W(m, m, 0.0);
    for (idx i = 0; i < m; i++) {
        for (idx j = 0; j < m; j++) {
            W(i, j) = (c[5] * B2(i, j)) + (c[3] * B(i, j));
        }
        W(i, i) += c[1];
    }

    mat<real> U(m, m, 0.0);
    matmul(as, W, U);

    mat<real> vp_u(m, m, 0.0);
    mat<real> vm_u(m, m, 0.0);
    // V_plus <- V + U, V_minus <- V - U.
    kernel::axpbyz(vp_u.data(), v_mat.data(), U.data(), real(1), real(1), m * m);
    kernel::axpbyz(vm_u.data(), v_mat.data(), U.data(), real(1), real(-1), m * m);

    // vm_u is constructed m-by-m.
    lu_result<real> fac = lu(vm_u);
    mat<real> E(m, m, 0.0);
    solve(fac, vp_u, E);

    for (int i = 0; i < s; i++) {
        mat<real> E2(m, m, 0.0);
        matmul(E, E, E2);
        E = std::move(E2);
    }

    return E;
}

struct krylov_step {
    vec<real> value;
    real error;
};

template <class Op>
requires linear_operator<Op, vec<real>, vec<real>> krylov_step
projected_expv(real t, const Op &A, const vec<real> &v, int dimension) {
    const real beta = norm(v);
    if (beta == 0.0)
        return {vec<real>(v.size(), 0.0), 0.0};

    array<vec<real>> basis;
    basis.reserve(dimension + 1);
    basis.push_back(v);
    scale(basis.back(), real(1) / beta);

    mat<real> H(dimension + 2, dimension + 2, 0.0);
    array<real> column(dimension + 1, 0.0);
    int size = dimension;
    bool breakdown = false;

    for (int j = 0; j < dimension; ++j) {
        vec<real> image(A.rows(), 0.0);
        A.apply(basis[j], image);
        const real image_norm = norm(image);
        const real next = dispatch::subspace::mgs_orthogonalize(basis, image, column, j + 1);
        for (int i = 0; i <= j; ++i)
            H(i, j) = column[i];
        H(j + 1, j) = next;

        if (next <= std::numeric_limits<real>::epsilon() * image_norm) {
            size = j + 1;
            breakdown = true;
            break;
        }
        scale(image, real(1) / next);
        basis.push_back(std::move(image));
    }

    real image_norm = 0.0;
    int projected_size = size;
    if (!breakdown) {
        vec<real> image(A.rows(), 0.0);
        A.apply(basis[size], image);
        image_norm = norm(image);
        H(size + 1, size) = 1.0;
        projected_size += 2;
    }

    mat<real> projected(projected_size, projected_size, 0.0);
    for (int i = 0; i < projected_size; ++i)
        for (int j = 0; j < projected_size; ++j)
            projected(i, j) = t * H(i, j);
    mat<real> exponential = dense_expm_pade6(projected);

    vec<real> value(v.size(), 0.0);
    for (int j = 0; j < size; ++j)
        axpy(beta * exponential(j, 0), basis[j], value);

    if (breakdown)
        return {std::move(value), 0.0};

    const real first = beta * std::abs(exponential(size, 0));
    const real second = beta * image_norm * std::abs(exponential(size + 1, 0));
    real error = first;
    if (first > real(10) * second)
        error = second;
    else if (first > second)
        error = first * second / (first - second);
    return {std::move(value), error};
}

} // namespace detail

inline vec<real> expv(real t, const spmat &A, const vec<real> &v, int m_max, real tol) {
    operators::sparse_op op(A);
    return expv(t, op, v, m_max, tol);
}

struct expv_options {
    int krylov_dimension = 30;
    real tolerance = 1e-8;
    int maximum_steps = 10000;
};

/// @brief Compute exp(tA)v with adaptive restarted Arnoldi steps.
template <class Op>
requires linear_operator<Op, vec<real>, vec<real>> vec<real>
restarted_expv(real t, const Op &A, const vec<real> &v, expv_options options = {}) {
    if (options.krylov_dimension <= 0)
        throw std::invalid_argument("restarted_expv: Krylov dimension must be positive");
    if (options.tolerance <= 0.0)
        throw std::invalid_argument("restarted_expv: tolerance must be positive");
    if (options.maximum_steps <= 0)
        throw std::invalid_argument("restarted_expv: maximum steps must be positive");
    if (t == 0.0)
        return v;

    vec<real> current = v;
    const real direction = std::copysign(1.0, t);
    const real target = std::abs(t);
    real advanced = 0.0;
    auto local_step = [&] {
        vec<real> image(A.rows(), 0.0);
        A.apply(current, image);
        real rate = norm(image) / std::max(norm(current), std::numeric_limits<real>::min());
        real krylov_scale = std::max(real(1), real(options.krylov_dimension) * real(0.5));
        return rate > 0.0 ? std::min(target - advanced, krylov_scale / rate)
                          : target - advanced;
    };
    real step = local_step();

    for (int attempt = 0; attempt < options.maximum_steps; ++attempt) {
        step = std::min(step, target - advanced);
        const real signed_step = direction * step;
        auto trial = detail::projected_expv(signed_step, A, current,
                                            options.krylov_dimension);
        const real error = trial.error;
        const real scale = std::max(real(1), norm(trial.value));
        const real threshold = std::max(options.tolerance * (step / target) * scale,
                                        real(32) * std::numeric_limits<real>::epsilon() * scale);
        if (std::isfinite(error) && error <= threshold) {
            current = std::move(trial.value);
            advanced += step;
            if (advanced >= target)
                return current;
            step = std::min(step * (error <= threshold * real(0.125) ? real(2) : real(1)),
                            local_step());
        } else {
            step *= real(0.5);
            if (step <= std::numeric_limits<real>::epsilon() * target) {
                std::ostringstream message;
                message << "restarted_expv: step underflow at t=" << advanced
                        << ", step=" << step << ", error=" << error
                        << ", tolerance=" << threshold;
                throw std::runtime_error(message.str());
            }
        }
    }
    throw std::runtime_error("restarted_expv: maximum steps exceeded");
}

inline vec<real> restarted_expv(real t, const spmat &A, const vec<real> &v,
                                expv_options options = {}) {
    operators::sparse_op op(A);
    return restarted_expv(t, op, v, options);
}

} // namespace num
