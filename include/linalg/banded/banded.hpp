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
  BandedMatrix(idx n, idx kl, idx ku);

  BandedMatrix(idx n, idx kl, idx ku, real val);

  ~BandedMatrix();

  BandedMatrix(const BandedMatrix&);
  BandedMatrix(BandedMatrix&&) noexcept;
  BandedMatrix& operator=(const BandedMatrix&);
  BandedMatrix& operator=(BandedMatrix&&) noexcept;

  [[nodiscard]] idx size() const { return n_; }
  [[nodiscard]] idx rows() const { return n_; }
  [[nodiscard]] idx cols() const { return n_; }

  [[nodiscard]] idx kl() const { return kl_; }

  [[nodiscard]] idx ku() const { return ku_; }

  [[nodiscard]] idx bandwidth() const { return kl_ + ku_ + 1; }

  [[nodiscard]] idx ldab() const { return ldab_; }

  real& operator()(idx i, idx j);
  real operator()(idx i, idx j) const;

  real& band(idx band_row, idx col);
  [[nodiscard]] real band(idx band_row, idx col) const;

  real* data() { return data_.get(); }
  [[nodiscard]] const real* data() const { return data_.get(); }

  [[nodiscard]] bool in_band(idx i, idx j) const;

  void to_gpu();
  void to_cpu();
  real* gpu_data() { return d_data_; }
  [[nodiscard]] const real* gpu_data() const { return d_data_; }
  [[nodiscard]] bool on_gpu() const { return d_data_ != nullptr; }

private:
  idx n_ = 0;
  idx kl_ = 0;
  idx ku_ = 0;
  idx ldab_ = 0;
  std::unique_ptr<real[]> data_;
  real* d_data_ = nullptr;
};

struct BandedSolverResult {
  bool success = false;
  idx pivot_row = 0;
  real rcond = 0.0;
};

/// @brief In-place banded \f$PA=LU\f$ factorization.
BandedSolverResult banded_lu(BandedMatrix& A, idx* ipiv);

/// @brief Solve \f$Ax=b\f$ using a precomputed banded LU factorization.
void banded_lu_solve(const BandedMatrix& A, const idx* ipiv, Vector& b);

/// @brief Solve \f$AX=B\f$ using a precomputed banded LU factorization.
void banded_lu_solve_multi(const BandedMatrix& A, const idx* ipiv, real* B, idx nrhs);

/// @brief Factor and solve \f$Ax=b\f$.
BandedSolverResult banded_solve(const BandedMatrix& A, const Vector& b, Vector& x);

/// @brief Compute \f$y=Ax\f$.
void banded_matvec(const BandedMatrix& A,
                   const Vector& x,
                   Vector& y,
                   Backend backend = default_backend);

/// @brief Compute \f$y=\alpha Ax+\beta y\f$.
void banded_gemv(real alpha,
                 const BandedMatrix& A,
                 const Vector& x,
                 real beta,
                 Vector& y,
                 Backend backend = default_backend);

/// @brief Estimate \f$1/\kappa_1(A)\f$.
real banded_rcond(const BandedMatrix& A, const idx* ipiv, real anorm);

/// @brief Compute \f$\|A\|_1\f$.
real banded_norm1(const BandedMatrix& A);

} // namespace num
