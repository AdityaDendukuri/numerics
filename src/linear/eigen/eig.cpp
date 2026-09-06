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

namespace seq {
eigen_result eig_sym(const mat &A_in, real tol, idx max_sweeps) {
    if (A_in.rows() != A_in.cols()) {
        throw std::invalid_argument("eig_sym: matrix must be square");
    }

    constexpr real rotation_tol = 1e-15;
    idx n = A_in.rows();
    mat A = A_in;
    mat V(n, n, 0.0);
    for (idx i = 0; i < n; ++i) {
        V(i, i) = 1.0;
    }

    idx sweeps = 0;
    bool converged = false;

    for (idx sweep = 0; sweep < max_sweeps; ++sweep) {
        real off = 0;
        for (idx p = 0; p < n; ++p) {
            for (idx q = p + 1; q < n; ++q) {
                off += A(p, q) * A(p, q);
            }
        }

        if (std::sqrt(2.0 * off) < tol) {
            converged = true;
            break;
        }

        for (idx p = 0; p < n - 1; ++p) {
            for (idx q = p + 1; q < n; ++q) {
                real apq = A(p, q);
                if (std::abs(apq) < rotation_tol) {
                    continue;
                }

                real app = A(p, p), aqq = A(q, q);
                real c = 1.0, s = 0.0;
                kernel::jacobi_rotation(app, aqq, apq, c, s);

                A(p, p) = (c * c * app) - (2.0 * c * s * apq) + (s * s * aqq);
                A(q, q) = (s * s * app) + (2.0 * c * s * apq) + (c * c * aqq);
                A(p, q) = A(q, p) = 0.0;

                for (idx r = 0; r < n; ++r) {
                    if (r == p || r == q) {
                        continue;
                    }
                    real arp = A(r, p), arq = A(r, q);
                    A(r, p) = A(p, r) = (c * arp) - (s * arq);
                    A(r, q) = A(q, r) = (s * arp) + (c * arq);
                }

                for (idx r = 0; r < n; ++r) {
                    real vrp = V(r, p), vrq = V(r, q);
                    V(r, p) = (c * vrp) - (s * vrq);
                    V(r, q) = (s * vrp) + (c * vrq);
                }
            }
        }
        ++sweeps;
    }

    vec values(n);
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
} // namespace seq

namespace omp {
eigen_result eig_sym(const mat &A_in, real tol, idx max_sweeps) {
    if (A_in.rows() != A_in.cols()) {
        throw std::invalid_argument("eig_sym: matrix must be square");
    }

    constexpr real rotation_tol = 1e-15;
    idx n = A_in.rows();
    mat A = A_in;
    mat V(n, n, 0.0);
    for (idx i = 0; i < n; ++i) {
        V(i, i) = 1.0;
    }

    idx sweeps = 0;
    bool converged = false;

    for (idx sweep = 0; sweep < max_sweeps; ++sweep) {
        real off = 0;
        for (idx p = 0; p < n; ++p) {
            for (idx q = p + 1; q < n; ++q) {
                off += A(p, q) * A(p, q);
            }
        }

        if (std::sqrt(2.0 * off) < tol) {
            converged = true;
            break;
        }

        for (idx p = 0; p < n - 1; ++p) {
            for (idx q = p + 1; q < n; ++q) {
                real apq = A(p, q);
                if (std::abs(apq) < rotation_tol) {
                    continue;
                }

                real app = A(p, p), aqq = A(q, q);
                real tau = (aqq - app) / (2.0 * apq);
                real t = std::copysign(1.0, tau) / (std::abs(tau) + std::sqrt(1.0 + (tau * tau)));
                real c = 1.0 / std::sqrt(1.0 + (t * t));
                real s = c * t;

                A(p, p) = (c * c * app) - (2.0 * c * s * apq) + (s * s * aqq);
                A(q, q) = (s * s * app) + (2.0 * c * s * apq) + (c * c * aqq);
                A(p, q) = A(q, p) = 0.0;

#ifdef NUMERICS_HAS_OMP
#pragma omp parallel for schedule(static) if (n >= 128)
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
#pragma omp parallel for schedule(static) if (n >= 128)
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

    vec values(n);
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
} // namespace omp

namespace lapack {
eigen_result eig_sym(const mat &A) {
#if defined(NUMERICS_HAS_LAPACK)
    if (A.rows() != A.cols()) {
        throw std::invalid_argument("eig_sym: matrix must be square");
    }
    idx n = A.rows();
    mat Aw = A;
    vec w(n);
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

eigen_result eig_sym(const linear::sym_mat<mat> &A, real tol, idx max_sweeps) {
#if defined(NUMERICS_HAS_LAPACK)
    return lapack::eig_sym(A.base());
#elif defined(NUMERICS_HAS_OMP)
    return omp::eig_sym(A.base(), tol, max_sweeps);
#else
    return seq::eig_sym(A.base(), tol, max_sweeps);
#endif
}

namespace unsafe {

eigen_result eig_sym(const mat &A, real tol, idx max_sweeps) {
    return num::eig_sym(linear::sym_mat<mat>(A), tol, max_sweeps);
}

} // namespace unsafe

} // namespace num
