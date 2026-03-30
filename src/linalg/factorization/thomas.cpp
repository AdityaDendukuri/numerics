/// @file linalg/factorization/thomas.cpp
/// @brief Thomas tridiagonal solver dispatcher.
///
/// Backend routing:
///   Backend::lapack  -> backends::lapack::thomas  (LAPACKE_dgtsv with pivoting)
///   Backend::gpu     -> CUDA batched Thomas kernel
///   everything else  -> backends::seq::thomas     (forward elimination + back sub)

#include "linalg/factorization/thomas.hpp"
#include "core/parallel/cuda_ops.hpp"
#include "backends/seq/impl.hpp"
#include "backends/lapack/impl.hpp"
#include <stdexcept>

namespace num {

void thomas(const Vector& a, const Vector& b, const Vector& c,
            const Vector& d, Vector& x, Backend backend) {
    idx n = b.size();
    if (a.size() != n - 1 || c.size() != n - 1 || d.size() != n || x.size() != n)
        throw std::invalid_argument("Dimension mismatch in Thomas solver");

    switch (backend) {
    case Backend::lapack:
        backends::lapack::thomas(a, b, c, d, x);
        return;
    case Backend::gpu:
#ifdef NUMERICS_HAS_CUDA
        {
            Vector ag = a; ag.to_gpu();
            Vector bg = b; bg.to_gpu();
            Vector cg = c; cg.to_gpu();
            Vector dg = d; dg.to_gpu();
            x = Vector(n);
            x.to_gpu();
            cuda::thomas_batched(ag.gpu_data(), bg.gpu_data(), cg.gpu_data(),
                                 dg.gpu_data(), x.gpu_data(), n, 1);
            x.to_cpu();
            return;
        }
#endif
        [[fallthrough]];
    default:
        backends::seq::thomas(a, b, c, d, x);
        return;
    }
}

} // namespace num
