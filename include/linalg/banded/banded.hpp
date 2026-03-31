/// @file banded.hpp
/// @brief High-performance banded matrix solver for HPC applications
///
/// Implements LAPACK-style banded storage and LU factorization with partial
/// pivoting. Optimized for vectorization and cache efficiency on modern
/// supercomputers (e.g., NCAR Derecho).
#pragma once

#include "core/types.hpp"
#include "core/vector.hpp"
#include "core/policy.hpp"
#include <memory>

namespace num {

/// @brief Banded matrix with efficient storage for direct solvers
///
/// Uses LAPACK-style band storage format optimized for LU factorization.
/// For a matrix with kl lower diagonals and ku upper diagonals, storage
/// layout is:
///
///   - Rows 0 to kl-1: Extra space for fill-in during LU factorization
///   - Rows kl to kl+ku: Upper diagonals (ku at top, main diagonal at kl+ku)
///   - Rows kl+ku+1 to 2*kl+ku: Lower diagonals
///
/// Element A(i,j) is stored at band(kl + ku + i - j, j) for max(0,j-ku) <= i <=
/// min(n-1,j+kl)
///
/// This format enables:
///   - Column-major access patterns for LU factorization
///   - SIMD-friendly memory layout
///   - Direct compatibility with LAPACK routines if needed
class BandedMatrix {
  public:
    /// @brief Construct a banded matrix
    /// @param n Matrix dimension (n x n)
    /// @param kl Number of lower diagonals
    /// @param ku Number of upper diagonals
    BandedMatrix(idx n, idx kl, idx ku);

    /// @brief Construct with initial value
    BandedMatrix(idx n, idx kl, idx ku, real val);

    ~BandedMatrix();

    // Rule of Five
    BandedMatrix(const BandedMatrix&);
    BandedMatrix(BandedMatrix&&) noexcept;
    BandedMatrix& operator=(const BandedMatrix&);
    BandedMatrix& operator=(BandedMatrix&&) noexcept;

    /// @brief Matrix dimension
    idx size() const {
        return n_;
    }
    idx rows() const {
        return n_;
    }
    idx cols() const {
        return n_;
    }

    /// @brief Number of lower diagonals
    idx kl() const {
        return kl_;
    }

    /// @brief Number of upper diagonals
    idx ku() const {
        return ku_;
    }

    /// @brief Total bandwidth (kl + ku + 1)
    idx bandwidth() const {
        return kl_ + ku_ + 1;
    }

    /// @brief Leading dimension of band storage (2*kl + ku + 1)
    idx ldab() const {
        return ldab_;
    }

    /// @brief Access element at (row, col) in original matrix coordinates
    /// @param i Row index (0-based)
    /// @param j Column index (0-based)
    /// @return Reference to element (undefined if outside band)
    real& operator()(idx i, idx j);
    real operator()(idx i, idx j) const;

    /// @brief Direct access to band storage
    /// @param band_row Row in band storage (0 to ldab-1)
    /// @param col Column index (0 to n-1)
    real& band(idx band_row, idx col);
    real band(idx band_row, idx col) const;

    /// @brief Raw pointer to band storage (column-major)
    real* data() {
        return data_.get();
    }
    const real* data() const {
        return data_.get();
    }

    /// @brief Check if (i,j) is within the band
    bool in_band(idx i, idx j) const;

    // GPU support
    void to_gpu();
    void to_cpu();
    real* gpu_data() {
        return d_data_;
    }
    const real* gpu_data() const {
        return d_data_;
    }
    bool on_gpu() const {
        return d_data_ != nullptr;
    }

  private:
    idx                     n_    = 0; ///< Matrix dimension
    idx                     kl_   = 0; ///< Number of lower diagonals
    idx                     ku_   = 0; ///< Number of upper diagonals
    idx                     ldab_ = 0; ///< Leading dimension (2*kl + ku + 1)
    std::unique_ptr<real[]> data_;     ///< Band storage (ldab * n elements)
    real*                   d_data_ = nullptr; ///< GPU data pointer
};

/// @brief Result from banded solver
struct BandedSolverResult {
    bool success   = false; ///< True if solve succeeded
    idx  pivot_row = 0;     ///< Row of zero pivot if singular (0 if success)
    real rcond =
        0.0; ///< Reciprocal condition number estimate (0 if not computed)
};

/// @brief LU factorization of banded matrix with partial pivoting
///
/// Computes A = P * L * U where:
///   - P is a permutation matrix
///   - L is lower triangular with unit diagonal
///   - U is upper triangular
///
/// The factorization is performed in-place. After factorization:
///   - U is stored in rows 0 to kl+ku (including fill-in)
///   - L multipliers are stored in rows kl+ku+1 to 2*kl+ku
///   - ipiv contains the pivot indices
///
/// @param A Banded matrix (modified in place with LU factors)
/// @param ipiv Pivot indices (size n, allocated by caller)
/// @return Result indicating success or location of zero pivot
///
/// Complexity: O(n * kl * (kl + ku)) operations
BandedSolverResult banded_lu(BandedMatrix& A, idx* ipiv);

/// @brief Solve banded system using precomputed LU factorization
///
/// Solves A*x = b where A has been factored by banded_lu().
/// The solution overwrites the right-hand side vector.
///
/// @param A LU-factored banded matrix from banded_lu()
/// @param ipiv Pivot indices from banded_lu()
/// @param b Right-hand side (overwritten with solution)
///
/// Complexity: O(n * (kl + ku)) operations
void banded_lu_solve(const BandedMatrix& A, const idx* ipiv, Vector& b);

/// @brief Solve multiple right-hand sides using LU factorization
///
/// Solves A*X = B where B has nrhs columns stored contiguously.
///
/// @param A LU-factored banded matrix
/// @param ipiv Pivot indices
/// @param B Right-hand sides (n x nrhs, column-major, overwritten with
/// solutions)
/// @param nrhs Number of right-hand sides
void banded_lu_solve_multi(const BandedMatrix& A,
                           const idx*          ipiv,
                           real*               B,
                           idx                 nrhs);

/// @brief Solve banded system Ax = b (convenience function)
///
/// Performs LU factorization and solve in one call. For solving multiple
/// systems with the same matrix, use banded_lu() and banded_lu_solve()
/// separately to avoid redundant factorization.
///
/// @param A Banded matrix (NOT modified - internal copy made)
/// @param b Right-hand side
/// @param x Solution vector (output)
/// @return Result indicating success or failure
BandedSolverResult banded_solve(const BandedMatrix& A,
                                const Vector&       b,
                                Vector&             x);

/// @brief Banded matrix-vector product y = A*x
///
/// Optimized for cache efficiency with loop ordering that accesses
/// the band storage in column-major order.
///
/// @param A Banded matrix
/// @param x Input vector
/// @param y Output vector (overwritten)
/// @param backend  Execution backend (Backend::gpu uses CUDA path when
/// available)
void banded_matvec(const BandedMatrix& A,
                   const Vector&       x,
                   Vector&             y,
                   Backend             backend = default_backend);

/// @brief Banded matrix-vector product y = alpha*A*x + beta*y
///
/// Generalized matrix-vector multiply with scaling factors.
///
/// @param alpha    Scalar multiplier for A*x
/// @param A        Banded matrix
/// @param x        Input vector
/// @param beta     Scalar multiplier for y
/// @param y        Input/output vector
/// @param backend  Execution backend (Backend::gpu uses CUDA path when
/// available)
void banded_gemv(real                alpha,
                 const BandedMatrix& A,
                 const Vector&       x,
                 real                beta,
                 Vector&             y,
                 Backend             backend = default_backend);

/// @brief Estimate reciprocal condition number of banded matrix
///
/// Uses 1-norm condition number estimation after LU factorization.
///
/// @param A LU-factored banded matrix
/// @param ipiv Pivot indices
/// @param anorm 1-norm of original matrix (before factorization)
/// @return Reciprocal condition number (small = ill-conditioned)
real banded_rcond(const BandedMatrix& A, const idx* ipiv, real anorm);

/// @brief Compute 1-norm of banded matrix
/// @param A Banded matrix
/// @return Maximum absolute column sum
real banded_norm1(const BandedMatrix& A);

} // namespace num
