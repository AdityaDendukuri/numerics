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
#include "container/parallel/lapack_wrapper.hpp"
#endif

#ifdef NUMERICS_HAS_CUDA
#include "container/parallel/cuda_ops.hpp"
#endif

namespace num {

namespace backends {

namespace seq {

template <typename Float = double>
inline void thomas(const BasicVector<Float> &a, const BasicVector<Float> &b,
                   const BasicVector<Float> &c, const BasicVector<Float> &d,
                   BasicVector<Float> &x) {
    const idx n = b.size();
    BasicVector<Float> b_work = b;
    BasicVector<Float> d_work = d;

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

inline void thomas(const Vector &a, const Vector &b, const Vector &c, const Vector &d, Vector &x) {
#if defined(NUMERICS_HAS_LAPACK)
    const idx n = b.size();
    std::vector<double> dl(a.data(), a.data() + (n - 1));
    std::vector<double> diag(b.data(), b.data() + n);
    std::vector<double> du(c.data(), c.data() + (n - 1));
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

} // namespace backends

/// Solve a tridiagonal system with lower a, diagonal b, upper c, and RHS d.
template <typename Float = double>
inline void thomas(const BasicVector<Float> &a, const BasicVector<Float> &b,
                   const BasicVector<Float> &c, const BasicVector<Float> &d,
                   BasicVector<Float> &x, Backend backend = backend::factor) {
    const idx n = b.size();
    if (a.size() != n - 1 || c.size() != n - 1 || d.size() != n || x.size() != n) {
        throw std::invalid_argument("Dimension mismatch in Thomas solver");
    }

    if constexpr (std::is_same_v<Float, double>) {
        switch (backend) {
        case backend::lapack:
            backends::lapack::thomas(a, b, c, d, x);
            return;
        case backend::gpu:
#ifdef NUMERICS_HAS_CUDA
        {
            Vector ag = a;
            ag.to_gpu();
            Vector bg = b;
            bg.to_gpu();
            Vector cg = c;
            cg.to_gpu();
            Vector dg = d;
            dg.to_gpu();
            x = Vector(n);
            x.to_gpu();
            cuda::thomas_batched(ag.gpu_data(), bg.gpu_data(), cg.gpu_data(), dg.gpu_data(),
                                 x.gpu_data(), n, 1);
            x.to_cpu();
            return;
        }
#endif
            [[fallthrough]];
        default:
            backends::seq::thomas(a, b, c, d, x);
            return;
        }
    } else {
        backends::seq::thomas(a, b, c, d, x);
    }
}

} // namespace num
