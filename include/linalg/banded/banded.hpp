/// @file banded.hpp
/// @brief Banded matrix storage and solvers.
#pragma once

#include "core/policy.hpp"
#include "core/types.hpp"
#include "core/vector.hpp"
#include <memory>

namespace num {

/// @brief LAPACK-style band storage.
///
/// Stores \f$A_{ij}\f$ at \f$\text{band}(k_l+k_u+i-j,j)\f$ when
/// \f$\max(0,j-k_u)\le i\le \min(n-1,j+k_l)\f$.
class BandedMatrix {
  public:
    /// Construct a zero-filled n-by-n matrix with lower/upper bandwidths kl/ku.
    BandedMatrix(idx n, idx kl, idx ku);

    /// Construct a banded matrix with every stored entry initialized to val.
    BandedMatrix(idx n, idx kl, idx ku, real val);

    ~BandedMatrix();

    BandedMatrix(const BandedMatrix &);
    BandedMatrix(BandedMatrix &&) noexcept;
    BandedMatrix &operator=(const BandedMatrix &);
    BandedMatrix &operator=(BandedMatrix &&) noexcept;

    /// Return the square matrix order.
    [[nodiscard]] idx size() const { return n_; }
    [[nodiscard]] idx rows() const { return n_; }
    [[nodiscard]] idx cols() const { return n_; }

    /// Return the lower bandwidth.
    [[nodiscard]] idx kl() const { return kl_; }

    /// Return the upper bandwidth.
    [[nodiscard]] idx ku() const { return ku_; }

    /// Return the number of mathematical diagonals in the band.
    [[nodiscard]] idx bandwidth() const { return kl_ + ku_ + 1; }

    /// Return the leading dimension of the LAPACK-compatible storage.
    [[nodiscard]] idx ldab() const { return ldab_; }

    /// Access a mathematical matrix entry inside the stored band.
    real &operator()(idx i, idx j);
    real operator()(idx i, idx j) const;

    /// Access an entry by physical band-storage coordinates.
    real &band(idx band_row, idx col);
    [[nodiscard]] real band(idx band_row, idx col) const;

    real *data() { return data_.get(); }
    [[nodiscard]] const real *data() const { return data_.get(); }

    /// Test whether mathematical entry (i,j) is explicitly stored.
    [[nodiscard]] bool in_band(idx i, idx j) const;

    void to_gpu();
    void to_cpu();
    real *gpu_data() { return d_data_; }
    [[nodiscard]] const real *gpu_data() const { return d_data_; }
    [[nodiscard]] bool on_gpu() const { return d_data_ != nullptr; }

  private:
    idx n_ = 0;
    idx kl_ = 0;
    idx ku_ = 0;
    idx ldab_ = 0;
    std::unique_ptr<real[]> data_;
    real *d_data_ = nullptr;
};

/// Status and diagnostics from a banded factorization or solve.
struct BandedSolverResult {
    bool success = false;
    idx pivot_row = 0;
    real rcond = 0.0;
};

/// @brief In-place banded \f$PA=LU\f$ factorization.
BandedSolverResult banded_lu(BandedMatrix &A, idx *ipiv);

/// @brief Solve \f$Ax=b\f$ using a precomputed banded LU factorization.
void banded_lu_solve(const BandedMatrix &A, const idx *ipiv, Vector &b);

/// @brief Solve \f$AX=B\f$ using a precomputed banded LU factorization.
void banded_lu_solve_multi(const BandedMatrix &A, const idx *ipiv, real *B, idx nrhs);

/// @brief Factor and solve \f$Ax=b\f$.
BandedSolverResult banded_solve(const BandedMatrix &A, const Vector &b, Vector &x);

/// @brief Compute \f$y=Ax\f$.
void banded_matvec(const BandedMatrix &A, const Vector &x, Vector &y,
                   Backend backend = default_backend);

/// @brief Compute \f$y=\alpha Ax+\beta y\f$.
void banded_gemv(real alpha, const BandedMatrix &A, const Vector &x, real beta, Vector &y,
                 Backend backend = default_backend);

/// @brief Estimate \f$1/\kappa_1(A)\f$.
real banded_rcond(const BandedMatrix &A, const idx *ipiv, real anorm);

/// @brief Compute \f$\|A\|_1\f$.
real banded_norm1(const BandedMatrix &A);

} // namespace num
