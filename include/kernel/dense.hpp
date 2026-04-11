/// @file kernel/dense.hpp
/// @brief Dense matrix inner kernels  (namespace num::kernel::dense)
///
/// These are the level-2 BLAS-equivalent operations exposed as first-class
/// kernels. They are the inner loops that factorization solvers and iterative
/// methods reduce to, but which are not currently accessible as standalone
/// functions in the rest of numerics.
///
///   ger(alpha, x, y, A)  -- A += alpha * x * y^T  (rank-1 update)
///   trsv_lower(L, b, x)  -- solve Lx = b  (L lower triangular)
///   trsv_upper(U, b, x)  -- solve Ux = b  (U upper triangular)
///
/// ger has seq_t / par_t overloads: the outer loop over rows is independent
/// and parallelizable.
///
/// trsv has no policy parameter: the outer loop has a serial dependency
/// (each row depends on all previous rows). BLAS cblas_dtrsv is used
/// automatically via raw:: when NUMERICS_HAS_BLAS is defined.
///
/// Include kernel/kernel.hpp to get all kernel sub-modules together.
#pragma once

#include "core/matrix.hpp"
#include "core/types.hpp"
#include "core/vector.hpp"
#include "kernel/policy.hpp"

namespace num::kernel::dense {

// ---------------------------------------------------------------------------
// ger: A[i,j] += alpha * x[i] * y[j]
// ---------------------------------------------------------------------------

/// @brief Sequential rank-1 update: calls raw::ger (routes to cblas_dger when
/// BLAS available; otherwise vectorizable row-update loop).
///
/// A must be (x.size() x y.size()) row-major.
void ger(real alpha, const Vector& x, const Vector& y, Matrix& A,
         seq_t) noexcept;

/// @brief Parallel rank-1 update: OMP parallel-for over rows.
/// Each row update A[i,:] += alpha*x[i]*y is independent.
void ger(real alpha, const Vector& x, const Vector& y, Matrix& A, par_t);

/// @brief Default policy
inline void ger(real alpha, const Vector& x, const Vector& y, Matrix& A) {
    ger(alpha, x, y, A, default_policy{});
}

// ---------------------------------------------------------------------------
// trsv_lower: solve Lx = b  (L lower triangular)
// ---------------------------------------------------------------------------

/// @brief Forward substitution: solve Lx = b.
///
/// L must be square. x is allocated to L.rows() if x.size() != L.rows().
/// No policy parameter: the outer loop has a serial dependency.
/// Dispatches to cblas_dtrsv via raw:: when NUMERICS_HAS_BLAS.
void trsv_lower(const Matrix& L, const Vector& b, Vector& x);

// ---------------------------------------------------------------------------
// trsv_upper: solve Ux = b  (U upper triangular)
// ---------------------------------------------------------------------------

/// @brief Back substitution: solve Ux = b.
///
/// U must be square. x is allocated to U.rows() if x.size() != U.rows().
/// No policy parameter: serial outer dependency.
/// Dispatches to cblas_dtrsv via raw:: when NUMERICS_HAS_BLAS.
void trsv_upper(const Matrix& U, const Vector& b, Vector& x);

} // namespace num::kernel::dense
