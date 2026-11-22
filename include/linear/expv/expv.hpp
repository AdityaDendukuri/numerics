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

mat<real> expm(mat<real> A);

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

    mat<real> E = expm(projected);

    vec<real> result(n, 0.0);
    for (int j = 0; j < m_actual; j++) {
        axpy(beta * E(j, 0), V[j], result);
    }

    return result;
}

vec<real> expv(real t, const spmat &A, const vec<real> &v, int m_max = 30, real tol = 1e-8);

namespace detail {

/// Parlett-Reinsch balancing: A <- D^-1 A D with D a diagonal of powers of 2. Returns D.
inline std::vector<real> balance(mat<real> &A) {
    const idx n = A.rows();
    std::vector<real> d(n, 1.0);
    bool converged = false;
    while (!converged) {
        converged = true;
        for (idx i = 0; i < n; ++i) {
            real c = 0;
            real r = 0;
            for (idx j = 0; j < n; ++j) {
                if (j != i) {
                    c += std::abs(A(j, i));
                    r += std::abs(A(i, j));
                }
            }
            if (c == 0 || r == 0) {
                continue;
            }
            real f = 1;
            const real s = c + r;
            while (c < r / 2) {
                f *= 2;
                c *= 4;
            }
            while (c > r * 2) {
                f /= 2;
                c /= 4;
            }
            if ((c + r) / f < 0.95 * s) {
                converged = false;
                d[i] *= f;
                for (idx j = 0; j < n; ++j) {
                    A(i, j) /= f;
                    A(j, i) *= f;
                }
            }
        }
    }
    return d;
}

} // namespace detail

/// @brief Dense matrix exponential: balancing, then scaling and squaring with the [13/13] Pade
/// approximant (Higham, SIMAX 26, 2005). Balancing and the larger Pade degree need fewer squarings,
/// which keeps graded, stiff matrices (such as Krylov projections of stiff generators) accurate.
inline mat<real> expm(mat<real> A) {
    const idx n = A.rows();
    const std::vector<real> d = detail::balance(A);
    real norm1 = 0;
    for (idx j = 0; j < n; ++j) {
        real s = 0;
        for (idx i = 0; i < n; ++i) {
            s += std::abs(A(i, j));
        }
        norm1 = num::max(norm1, s);
    }
    constexpr real theta13 = 5.371920351148152;
    const int s = norm1 > theta13 ? static_cast<int>(std::ceil(std::log2(norm1 / theta13))) : 0;
    kernel::scale(A.data(), std::ldexp(1.0, -s), n * n);
    static constexpr real b[14] = {64764752532480000.0, 32382376266240000.0, 7771770303897600.0,
                                   1187353796428800.0,  129060195264000.0,   10559470521600.0,
                                   670442572800.0,      33522128640.0,       1323241920.0,
                                   40840800.0,          960960.0,            16380.0,
                                   182.0,               1.0};
    auto mul = [n](const mat<real> &X, const mat<real> &Y) {
        mat<real> Z(n, n, 0.0);
        matmul(X, Y, Z);
        return Z;
    };
    const mat<real> A2 = mul(A, A), A4 = mul(A2, A2), A6 = mul(A4, A2);
    mat<real> T1(n, n, 0.0), T2(n, n, 0.0);
    for (idx k = 0; k < n * n; ++k) {
        T1.data()[k] = b[13] * A6.data()[k] + b[11] * A4.data()[k] + b[9] * A2.data()[k];
        T2.data()[k] = b[12] * A6.data()[k] + b[10] * A4.data()[k] + b[8] * A2.data()[k];
    }
    const mat<real> W1 = mul(A6, T1), W2 = mul(A6, T2);
    mat<real> U0(n, n, 0.0), V(n, n, 0.0);
    for (idx i = 0; i < n; ++i) {
        for (idx j = 0; j < n; ++j) {
            const real id = i == j ? 1.0 : 0.0;
            U0(i, j) = W1(i, j) + b[7] * A6(i, j) + b[5] * A4(i, j) + b[3] * A2(i, j) + b[1] * id;
            V(i, j) = W2(i, j) + b[6] * A6(i, j) + b[4] * A4(i, j) + b[2] * A2(i, j) + b[0] * id;
        }
    }
    const mat<real> U = mul(A, U0);
    mat<real> P(n, n, 0.0), Q(n, n, 0.0);
    for (idx k = 0; k < n * n; ++k) {
        P.data()[k] = V.data()[k] + U.data()[k];
        Q.data()[k] = V.data()[k] - U.data()[k];
    }
    mat<real> E(n, n, 0.0);
    solve(lu(Q), P, E);
    for (int k = 0; k < s; ++k) {
        E = mul(E, E);
    }
    for (idx i = 0; i < n; ++i) {  // undo balancing: exp(A) = D exp(D^-1 A D) D^-1
        for (idx j = 0; j < n; ++j) {
            E(i, j) *= d[i] / d[j];
        }
    }
    return E;
}

namespace detail {

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
    mat<real> exponential = expm(projected);

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
