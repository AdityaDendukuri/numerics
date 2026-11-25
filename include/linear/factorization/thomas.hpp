/// @file linear/factorization/thomas.hpp
/// @brief Thomas algorithm for tridiagonal systems.
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
inline void thomas(const vec<Float> &a, const vec<Float> &b,
                   const vec<Float> &c, const vec<Float> &d,
                   vec<Float> &x) {
    const idx n = b.size();
    vec<Float> b_work = b;
    vec<Float> d_work = d;

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

inline void thomas(const vec<real> &a, const vec<real> &b, const vec<real> &c, const vec<real> &d, vec<real> &x) {
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
inline void thomas(const vec<real> &a, const vec<real> &b, const vec<real> &c, const vec<real> &d, vec<real> &x) {
    const idx n = b.size();
    vec<real> ag = a;
    ag.to_gpu();
    vec<real> bg = b;
    bg.to_gpu();
    vec<real> cg = c;
    cg.to_gpu();
    vec<real> dg = d;
    dg.to_gpu();
    x = vec<real>(n);
    x.to_gpu();
    num::cuda::thomas_batched(ag.gpu_data(), bg.gpu_data(), cg.gpu_data(), dg.gpu_data(),
                              x.gpu_data(), n, 1);
    x.to_cpu();
}
} // namespace cuda
#endif

/// @brief Tridiagonal solve (Thomas): \f$a_{i-1} x_{i-1} + b_i x_i + c_i x_{i+1} = d_i\f$.
/// Uses LAPACK `dgtsv` when available.
/// @throws std::invalid_argument If the diagonal lengths are inconsistent.
template <typename Float = double>
inline void thomas(const vec<Float> &a, const vec<Float> &b,
                   const vec<Float> &c, const vec<Float> &d,
                   vec<Float> &x) {
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
