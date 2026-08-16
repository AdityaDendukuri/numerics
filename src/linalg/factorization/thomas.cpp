/// @file linalg/factorization/thomas.cpp
/// @brief Thomas tridiagonal solver dispatcher + implementations (seq, lapack, gpu).

#include "linalg/factorization/thomas.hpp"
#include "core/parallel/cuda_ops.hpp"
#include <stdexcept>
#include <string>
#include <vector>

#if defined(NUMERICS_HAS_LAPACK)
    #include "core/parallel/lapack_wrapper.hpp"
#endif


namespace num {

namespace backends {

namespace seq {
void thomas(const Vector& a, const Vector& b, const Vector& c, const Vector& d, Vector& x) {
    const idx n = b.size();
    Vector b_work = b;
    Vector d_work = d;

    for (idx i = 1; i < n; ++i) {
        real w = a[i - 1] / b_work[i - 1];
        b_work[i] -= w * c[i - 1];
        d_work[i] -= w * d_work[i - 1];
    }

    x[n - 1] = d_work[n - 1] / b_work[n - 1];
    for (idx i = n - 1; i > 0; --i)
        x[i - 1] = (d_work[i - 1] - c[i - 1] * x[i]) / b_work[i - 1];
}
} // namespace seq

namespace lapack {
void thomas(const Vector& a, const Vector& b, const Vector& c, const Vector& d, Vector& x) {
#if defined(NUMERICS_HAS_LAPACK)
    const idx n = b.size();
    std::vector<double> dl(a.data(), a.data() + (n - 1));
    std::vector<double> diag(b.data(), b.data() + n);
    std::vector<double> du(c.data(), c.data() + (n - 1));
    x = d;
    int info = LAPACKE_dgtsv(LAPACK_ROW_MAJOR,
                             static_cast<lapack_int>(n),
                             1,
                             dl.data(),
                             diag.data(),
                             du.data(),
                             x.data(),
                             1);
    if (info != 0)
        throw std::runtime_error("thomas (lapack): dgtsv failed, info=" + std::to_string(info));
#else
    seq::thomas(a, b, c, d, x);
#endif
}
} // namespace lapack

} // namespace backends

void thomas(const Vector& a,
            const Vector& b,
            const Vector& c,
            const Vector& d,
            Vector& x,
            Backend backend) {
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
            cuda::thomas_batched(ag.gpu_data(),
                                 bg.gpu_data(),
                                 cg.gpu_data(),
                                 dg.gpu_data(),
                                 x.gpu_data(),
                                 n,
                                 1);
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
