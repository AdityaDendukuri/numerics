/// @file kernel/subspace.cpp
/// @brief Implementations for num::kernel::subspace.

#include "kernel/subspace.hpp"
#include "core/matrix.hpp"
#include "core/vector.hpp"
#include "linalg/sparse/sparse.hpp"
#ifdef NUMERICS_HAS_BLAS
#include <cblas.h>
#endif

namespace num::kernel::subspace {

// ---------------------------------------------------------------------------
// LinearOp concrete implementations
// ---------------------------------------------------------------------------

void DenseOp::apply(const Vector& x, Vector& y) const {
    if (y.size() != A_.rows()) { y = Vector(A_.rows()); }
    matvec(A_, x, y, b_);
}

void SparseOp::apply(const Vector& x, Vector& y) const {
    if (y.size() != A_.n_rows()) { y = Vector(A_.n_rows()); }
    sparse_matvec(A_, x, y);
}

// ---------------------------------------------------------------------------
// mgs_orthogonalize  (vector-basis overload)
// ---------------------------------------------------------------------------

real mgs_orthogonalize(const std::vector<Vector>& basis,
                       Vector&                    v,
                       std::vector<real>&         h,
                       idx                        k) {
    for (idx i = 0; i < k; ++i) {
        h[i] = dot(v, basis[i]);
        axpy(-h[i], basis[i], v);
    }
    return norm(v);
}

// ---------------------------------------------------------------------------
// mgs_orthogonalize  (matrix-basis overload, strided BLAS when available)
// ---------------------------------------------------------------------------

real mgs_orthogonalize(const Matrix& basis, idx k, Vector& v) {
    const idx n      = basis.rows();
    const idx stride = basis.cols(); // row-major: stride between column elements

    for (idx l = 0; l < k; ++l) {
#ifdef NUMERICS_HAS_BLAS
        const real proj = cblas_ddot(static_cast<int>(n),
                                     basis.data() + l,
                                     static_cast<int>(stride),
                                     v.data(), 1);
        cblas_daxpy(static_cast<int>(n), -proj,
                    basis.data() + l, static_cast<int>(stride),
                    v.data(), 1);
#else
        real proj = 0.0;
        for (idx i = 0; i < n; ++i) { proj += basis(i, l) * v[i]; }
        for (idx i = 0; i < n; ++i) { v[i] -= proj * basis(i, l); }
#endif
    }
    return norm(v);
}

// ---------------------------------------------------------------------------
// arnoldi_step
// ---------------------------------------------------------------------------

real arnoldi_step(const LinearOp&      A,
                  std::vector<Vector>& basis,
                  std::vector<real>&   h,
                  idx                  k,
                  Vector&              scratch,
                  real                 breakdown_tol) {
    // Step 1: w = A * basis[k]  (written into the pre-allocated scratch buffer)
    A.apply(basis[k], scratch);

    // Step 2 & 3: MGS against basis[0..k], fills h[0..k]; returns ||w||
    const real beta = mgs_orthogonalize(basis, scratch, h, k + 1);
    h[k + 1] = beta;

    // Step 4: normalize and extend basis (skip on lucky breakdown)
    if (beta > breakdown_tol) {
        scale(scratch, real(1) / beta);
        basis.push_back(scratch);
    }

    return beta;
}

} // namespace num::kernel::subspace
