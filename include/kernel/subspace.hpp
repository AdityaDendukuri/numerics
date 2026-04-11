/// @file kernel/subspace.hpp
/// @brief Subspace construction and orthogonalization kernels.
///                                                        (namespace num::kernel::subspace)
///
/// These are the inner loops that any algorithm building or maintaining an
/// orthonormal basis reduces to -- Krylov solvers (GMRES, MINRES), Lanczos,
/// rational Krylov, exponential integrators (expv), randomized SVD, POD,
/// and model-order reduction all share the same core operations.
///
/// Linear operator interface:
///   LinearOp            -- abstract base: apply(x, y), rows(), cols()
///   DenseOp             -- wraps Matrix  (Backend-dispatched matvec)
///   SparseOp            -- wraps SparseMatrix
///   CallableOp<F>       -- wraps any callable void(const Vector&, Vector&)
///   make_op(f, n)       -- factory returning CallableOp<F> (stack, typed)
///   make_op_ptr(f, n)   -- factory returning unique_ptr<LinearOp> (heap, erased)
///
/// Orthogonalization kernels:
///   mgs_orthogonalize   -- Modified Gram-Schmidt: orthogonalize one vector
///                          against an existing basis, sequential and stable.
///   arnoldi_step        -- One Arnoldi step: apply operator + MGS + normalize.
///                          The inner loop that GMRES, Lanczos, and expv share.
///
/// Key design choice -- scratch buffer:
///   arnoldi_step takes a pre-allocated scratch Vector for the operator apply.
///   Callers allocate it once before the loop; the kernel reuses it at every
///   step. This avoids one heap allocation per Arnoldi step.
///
/// Key design choice -- no OMP inside subspace kernels:
///   mgs_orthogonalize and arnoldi_step call num::dot / num::axpy / num::norm
///   with default_backend (which is BLAS > SIMD > blocked). The outer basis-
///   expansion loop is inherently serial (each step depends on the previous).
///   OMP at this level would only cause nested-parallelism issues.
#pragma once

#include "core/matrix.hpp"
#include "core/policy.hpp"
#include "core/types.hpp"
#include "core/vector.hpp"
#include "linalg/sparse/sparse.hpp"
#include <memory>
#include <vector>

namespace num::kernel::subspace {

// ===========================================================================
// Linear operator interface
// ===========================================================================

/// @brief Abstract matrix-free linear operator: y = A*x.
///
/// Subclass this or use DenseOp / SparseOp / CallableOp / make_op.
/// Virtual dispatch overhead is O(1) per apply(); work inside is O(n) or
/// O(nnz), so the overhead is negligible.
struct LinearOp {
    virtual ~LinearOp() = default;

    /// @brief y = A*x  (y must be pre-allocated to the correct size)
    virtual void apply(const Vector& x, Vector& y) const = 0;

    [[nodiscard]] virtual idx rows() const noexcept = 0;
    [[nodiscard]] virtual idx cols() const noexcept = 0;
};

/// @brief Wrap a dense Matrix as a LinearOp.
///
/// The matvec is dispatched to the chosen Backend (default: default_backend).
struct DenseOp final : LinearOp {
    explicit DenseOp(const Matrix& A, Backend b = default_backend)
        : A_(A), b_(b) {}

    void apply(const Vector& x, Vector& y) const override;
    [[nodiscard]] idx rows() const noexcept override { return A_.rows(); }
    [[nodiscard]] idx cols() const noexcept override { return A_.cols(); }

private:
    const Matrix& A_;
    Backend       b_;
};

/// @brief Wrap a SparseMatrix as a LinearOp.
struct SparseOp final : LinearOp {
    explicit SparseOp(const SparseMatrix& A) : A_(A) {}

    void apply(const Vector& x, Vector& y) const override;
    [[nodiscard]] idx rows() const noexcept override { return A_.n_rows(); }
    [[nodiscard]] idx cols() const noexcept override { return A_.n_cols(); }

private:
    const SparseMatrix& A_;
};

/// @brief Wrap any callable void(const Vector&, Vector&) as a LinearOp.
///
/// Useful for structured operators -- stencils, FFT-based, implicit matrices --
/// without forming an explicit matrix. The functor is stored by value.
///
/// apply() pre-allocates y to the correct size (n_) before calling f_ so that
/// callers (e.g. arnoldi_step passing a pre-allocated scratch buffer) never
/// trigger a heap allocation inside the hot Arnoldi loop. Lambdas passed to
/// make_op / make_op_ptr can safely assume y.size() == n on entry.
///
/// @code
///   auto op = num::kernel::subspace::make_op(
///       [&](const Vector& x, Vector& y) { laplacian(x, y, N); }, N * N);
/// @endcode
template<typename F>
struct CallableOp final : LinearOp {
    CallableOp(F f, idx n) : f_(std::move(f)), n_(n) {}

    void apply(const Vector& x, Vector& y) const override {
        if (y.size() != n_) { y = Vector(n_); }
        f_(x, y);
    }
    [[nodiscard]] idx rows() const noexcept override { return n_; }
    [[nodiscard]] idx cols() const noexcept override { return n_; }

private:
    F   f_;
    idx n_;
};

/// @brief Factory: wrap a callable as a stack-allocated CallableOp<F>.
///
/// Use when the operator type is known at the call site (zero overhead;
/// compiler can inline apply()).
template<typename F>
[[nodiscard]] CallableOp<F> make_op(F f, idx n) {
    return CallableOp<F>(std::move(f), n);
}

/// @brief Factory: wrap a callable as a heap-allocated LinearOp.
///
/// Use when you need type-erased storage (unique_ptr<LinearOp>).
template<typename F>
[[nodiscard]] std::unique_ptr<LinearOp> make_op_ptr(F f, idx n) {
    return std::make_unique<CallableOp<F>>(std::move(f), n);
}

// ===========================================================================
// Orthogonalization kernels
// ===========================================================================

/// @brief Modified Gram-Schmidt: orthogonalize v against basis[0..k-1].
///
/// For each i in [0, k), computes h[i] = <v, basis[i]> then subtracts
/// h[i] * basis[i] from v. Updates v in-place.
///
/// h must be pre-allocated with size >= k.
///
/// Returns ||v||₂ after all projections are removed (= the next subdiagonal
/// element in the Hessenberg matrix when used inside arnoldi_step).
///
/// MGS performs the projections sequentially; each one modifies v before the
/// next dot product, giving better numerical stability than classical GS at
/// the cost of k sequential passes over v.
///
/// Uses num::dot and num::axpy with default_backend (BLAS > SIMD > seq).
[[nodiscard]] real mgs_orthogonalize(const std::vector<Vector>& basis,
                                     Vector&                    v,
                                     std::vector<real>&         h,
                                     idx                        k);

/// @brief Modified Gram-Schmidt: orthogonalize v against columns 0..k-1 of a
///        column-stored basis matrix.
///
/// Overload for algorithms (e.g. Lanczos) that store their Krylov basis
/// column-by-column in a dense Matrix rather than a std::vector<Vector>.
/// The Matrix is stored row-major, so column l is accessed with stride
/// basis.cols() between successive elements.
///
/// When NUMERICS_HAS_BLAS is defined, uses cblas_ddot / cblas_daxpy with
/// the correct stride so the BLAS handles the strided access natively.
/// Otherwise falls back to a plain loop via basis(i, l).
///
/// @param basis  Row-major Matrix of shape (n, max_steps); columns are the
///               orthonormal basis vectors.
/// @param k      Number of columns to project out (columns 0..k-1).
/// @param v      Vector to orthogonalize in-place (length n).
/// @return       ||v||₂ after all projections are removed.
[[nodiscard]] real mgs_orthogonalize(const Matrix& basis, idx k, Vector& v);

/// @brief One Arnoldi step: expand the orthonormal basis by one vector.
///
/// Algorithm:
///   1. w = A * basis[k]              (operator apply via scratch buffer)
///   2. h[0..k] = mgs_orthogonalize(basis[0..k], w)
///   3. h[k+1]  = ||w||              (subdiagonal Hessenberg element)
///   4. if h[k+1] > breakdown_tol:
///        w /= h[k+1]; basis.push_back(w)   (new orthonormal basis vector)
///
/// @param A             Linear operator
/// @param basis         Current orthonormal basis (basis[0..k] present on entry;
///                      basis[k+1] appended on exit if no breakdown)
/// @param h             Hessenberg column k; must be pre-allocated with size >= k+2
/// @param k             Current step index (0-based)
/// @param scratch       Pre-allocated Vector of size n; reused across steps to
///                      avoid per-step heap allocation
/// @param breakdown_tol Lucky breakdown threshold (default 1e-14)
///
/// @return h[k+1]: the subdiagonal element. Near-zero signals lucky breakdown
///         (A-invariant subspace found; solution is exact).
[[nodiscard]] real arnoldi_step(const LinearOp&    A,
                                std::vector<Vector>& basis,
                                std::vector<real>&   h,
                                idx                  k,
                                Vector&              scratch,
                                real breakdown_tol = real(1e-14));

} // namespace num::kernel::subspace
