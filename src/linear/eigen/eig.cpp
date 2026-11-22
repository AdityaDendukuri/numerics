/// @file linear/eigen/eig.cpp
/// @brief Full symmetric eigendecomposition dispatcher + implementations (seq, omp,
/// lapack).

#include "linear/eigen/jacobi_eig.hpp"
#include "core/debug.hpp"
#include "kernel/kernel.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

#if defined(NUMERICS_HAS_LAPACK)
#include "lapack/lapack_wrapper.hpp"
#endif

#if defined(NUMERICS_HAS_OMP)
#include <omp.h>
#endif

namespace num {

// Cyclic Jacobi. With `parallel`, the two row updates of each rotation are threaded once n is
// large enough to pay for the fork; the arithmetic is the same either way.
static eigen_result jacobi_eig_sym(const mat<real> &A_in, real tol, idx max_sweeps, bool parallel) {
    if (A_in.rows() != A_in.cols()) {
        throw std::invalid_argument("eig_sym: matrix must be square");
    }

    idx n = A_in.rows();
    mat<real> A = A_in;
    mat<real> V(n, n, 0.0);
    for (idx i = 0; i < n; ++i) {
        V(i, i) = 1.0;
    }
    [[maybe_unused]] const bool threaded = parallel && n >= 128;

    // Rotations preserve the Frobenius norm, so it scales the stopping test throughout.
    real frobenius_sq = 0;
    for (idx i = 0; i < n * n; ++i) {
        frobenius_sq += A.data()[i] * A.data()[i];
    }
    const real frobenius = std::sqrt(frobenius_sq);

    idx sweeps = 0;
    bool converged = false;

    for (idx sweep = 0; sweep < max_sweeps; ++sweep) {
        real off = 0;
        for (idx p = 0; p < n; ++p) {
            for (idx q = p + 1; q < n; ++q) {
                off += A(p, q) * A(p, q);
            }
        }

        if (std::sqrt(2.0 * off) <= tol * frobenius) {
            converged = true;
            break;
        }

        for (idx p = 0; p < n - 1; ++p) {
            for (idx q = p + 1; q < n; ++q) {
                real apq = A(p, q);
                real app = A(p, p), aqq = A(q, q);
                // Negligible against its diagonal pair (Demmel and Veselic).
                if (std::abs(apq) <= num::unit_roundoff * std::sqrt(std::abs(app * aqq)) ||
                    apq == 0.0) {
                    continue;
                }

                real c = 1.0, s = 0.0;
                num::jacobi_rotation(app, aqq, apq, c, s);

                A(p, p) = (c * c * app) - (2.0 * c * s * apq) + (s * s * aqq);
                A(q, q) = (s * s * app) + (2.0 * c * s * apq) + (c * c * aqq);
                A(p, q) = A(q, p) = 0.0;

#ifdef NUMERICS_HAS_OMP
#pragma omp parallel for schedule(static) if (threaded)
#endif
                for (idx r = 0; r < n; ++r) {
                    if (r == p || r == q) {
                        continue;
                    }
                    real arp = A(r, p), arq = A(r, q);
                    A(r, p) = A(p, r) = (c * arp) - (s * arq);
                    A(r, q) = A(q, r) = (s * arp) + (c * arq);
                }

#ifdef NUMERICS_HAS_OMP
#pragma omp parallel for schedule(static) if (threaded)
#endif
                for (idx r = 0; r < n; ++r) {
                    real vrp = V(r, p), vrq = V(r, q);
                    V(r, p) = (c * vrp) - (s * vrq);
                    V(r, q) = (s * vrp) + (c * vrq);
                }
            }
        }
        ++sweeps;
    }

    vec<real> values(n);
    for (idx i = 0; i < n; ++i) {
        values[i] = A(i, i);
    }

    for (idx i = 0; i < n - 1; ++i) {
        idx min_j = i;
        for (idx j = i + 1; j < n; ++j) {
            if (values[j] < values[min_j]) {
                min_j = j;
            }
        }
        if (min_j != i) {
            std::swap(values[i], values[min_j]);
            for (idx r = 0; r < n; ++r) {
                std::swap(V(r, i), V(r, min_j));
            }
        }
    }

    return {values, V, sweeps, converged};
}

namespace seq {
eigen_result eig_sym(const mat<real> &A, real tol, idx max_sweeps) {
    return jacobi_eig_sym(A, tol, max_sweeps, false);
}
} // namespace seq

namespace omp {
eigen_result eig_sym(const mat<real> &A, real tol, idx max_sweeps) {
    return jacobi_eig_sym(A, tol, max_sweeps, true);
}
} // namespace omp

namespace lapack {
eigen_result eig_sym(const mat<real> &A) {
#if defined(NUMERICS_HAS_LAPACK)
    if (A.rows() != A.cols()) {
        throw std::invalid_argument("eig_sym: matrix must be square");
    }
    idx n = A.rows();
    mat<real> Aw = A;
    vec<real> w(n);
    int info = LAPACKE_dsyevd(LAPACK_ROW_MAJOR, 'V', 'U', static_cast<lapack_int>(n), Aw.data(),
                              static_cast<lapack_int>(n), w.data());
    if (info != 0) {
        throw std::runtime_error("eig_sym (lapack): dsyevd failed, info=" + std::to_string(info));
    }
    return {w, Aw, 0, true};
#else
    return seq::eig_sym(A, 1e-12, 100);
#endif
}
} // namespace lapack

eigen_result eig_sym(const with_law<mat<real>, law::self_adjoint> &A, real tol, idx max_sweeps) {
#if defined(NUMERICS_HAS_LAPACK)
    return lapack::eig_sym(A.base());
#elif defined(NUMERICS_HAS_OMP)
    return omp::eig_sym(A.base(), tol, max_sweeps);
#else
    return seq::eig_sym(A.base(), tol, max_sweeps);
#endif
}

namespace unsafe {

eigen_result eig_sym(const mat<real> &A, real tol, idx max_sweeps) {
    return num::eig_sym(with_law<mat<real>, law::self_adjoint>(A), tol, max_sweeps);
}

} // namespace unsafe

} // namespace num
