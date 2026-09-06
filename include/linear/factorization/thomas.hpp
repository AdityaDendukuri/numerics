/// @file linear/factorization/thomas.hpp
/// @brief Pure template Thomas algorithm for tridiagonal systems.
///
/// Solves \f$a_{i-1}x_{i-1}+b_i x_i+c_i x_{i+1}=d_i\f$ in \f$O(n)\f$.
#pragma once

#include "core/policy.hpp"
#include "core/types.hpp"
#include "container/vector.hpp"
#include <concepts>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(NUMERICS_HAS_LAPACK)
#include "lapack/lapack_wrapper.hpp"
#endif

#ifdef NUMERICS_HAS_CUDA
#include "cuda/cuda_ops.hpp"
#endif

namespace num {

namespace seq {

template <typename Float = double>
inline void thomas(const basic_vec<Float> &a, const basic_vec<Float> &b,
                   const basic_vec<Float> &c, const basic_vec<Float> &d,
                   basic_vec<Float> &x) {
    const idx n = b.size();
    basic_vec<Float> b_work = b;
    basic_vec<Float> d_work = d;

    for (idx i = 1; i < n; ++i) {
        Float w = a[i - 1] / b_work[i - 1];
        b_work[i] -= w * c[i - 1];
        d_work[i] -= w * d_work[i - 1];
    }

    x[n - 1] = d_work[n - 1] / b_work[n - 1];
    for (idx i = n - 1; i > 0; --i) {
        x[i - 1] = (d_work[i - 1] - (c[i - 1] * x[i])) / b_work[i - 1];
    }
}

} // namespace seq

namespace lapack {

inline void thomas(const vec &a, const vec &b, const vec &c, const vec &d, vec &x) {
#if defined(NUMERICS_HAS_LAPACK)
    const idx n = b.size();
    array<double> dl(a.data(), a.data() + (n - 1));
    array<double> diag(b.data(), b.data() + n);
    array<double> du(c.data(), c.data() + (n - 1));
    x = d;
    int info = LAPACKE_dgtsv(LAPACK_ROW_MAJOR, static_cast<lapack_int>(n), 1, dl.data(),
                             diag.data(), du.data(), x.data(), 1);
    if (info != 0) {
        throw std::runtime_error("thomas (lapack): dgtsv failed, info=" + std::to_string(info));
    }
#else
    seq::thomas(a, b, c, d, x);
#endif
}

} // namespace lapack

#ifdef NUMERICS_HAS_CUDA
namespace cuda {
inline void thomas(const vec &a, const vec &b, const vec &c, const vec &d, vec &x) {
    const idx n = b.size();
    vec ag = a;
    ag.to_gpu();
    vec bg = b;
    bg.to_gpu();
    vec cg = c;
    cg.to_gpu();
    vec dg = d;
    dg.to_gpu();
    x = vec(n);
    x.to_gpu();
    num::cuda::thomas_batched(ag.gpu_data(), bg.gpu_data(), cg.gpu_data(), dg.gpu_data(),
                              x.gpu_data(), n, 1);
    x.to_cpu();
}
} // namespace cuda
#endif

/// @brief Solve tridiagonal linear system \f$a_{i-1} x_{i-1} + b_i x_i + c_i x_{i+1} = d_i\f$ in \f$\mathcal{O}(n)\f$ time.
///
/// Executes the Thomas algorithm (specialized Gaussian elimination for tridiagonal systems) with zero heap allocations.
/// Picks LAPACK (`dgtsv`) if configured, else the in-tree sequential elimination.
/// To force a specific one (including the CUDA batched kernel when built with
/// CUDA), call `num::lapack::thomas`/`num::seq::thomas`/`num::cuda::thomas`.
///
/// @tparam Float Floating-point scalar type (`double`, `float`).
/// @param a Subdiagonal entries (\f$n-1\f$ elements: \f$a_0, \dots, a_{n-2}\f$).
/// @param b Main diagonal entries (\f$n\f$ elements: \f$b_0, \dots, b_{n-1}\f$).
/// @param c Superdiagonal entries (\f$n-1\f$ elements: \f$c_0, \dots, c_{n-2}\f$).
/// @param d Right-hand side vector (\f$n\f$ elements).
/// @param x Output solution vector (\f$n\f$ elements).
/// @throws std::invalid_argument If dimensions do not match (\f$a, c\f$ size \f$n-1\f$, \f$b, d, x\f$ size \f$n\f$).
/// @see banded_solve, lu_solve
template <typename Float = double>
inline void thomas(const basic_vec<Float> &a, const basic_vec<Float> &b,
                   const basic_vec<Float> &c, const basic_vec<Float> &d,
                   basic_vec<Float> &x) {
    const idx n = b.size();
    if (a.size() != n - 1 || c.size() != n - 1 || d.size() != n || x.size() != n) {
        throw std::invalid_argument("Dimension mismatch in Thomas solver");
    }

    if constexpr (std::is_same_v<Float, double>) {
#if defined(NUMERICS_HAS_LAPACK)
        lapack::thomas(a, b, c, d, x);
#else
        seq::thomas(a, b, c, d, x);
#endif
    } else {
        seq::thomas(a, b, c, d, x);
    }
}

} // namespace num
