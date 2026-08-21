/// @file factorization/tridiag_complex.hpp
/// @brief Precomputed Thomas solver for constant-coefficient complex systems.
///
/// Solves \f$a x_{k-1}+b x_k+c x_{k+1}=d_k\f$ with fixed \f$a,b,c\f$.
#pragma once

#include "core/types.hpp"
#include <complex>
#include <vector>

namespace num {

/// Reusable factorization of a constant-coefficient complex tridiagonal matrix.
struct ComplexTriDiag {
  using cplx = std::complex<double>;

  std::vector<cplx> c_mod;
  std::vector<cplx> inv_b;
  int n = 0;
  cplx a_coeff = {};

  /// Factor an n-by-n tridiagonal matrix with constant lower/diagonal/upper entries.
  void factor(int n_, cplx a_, cplx b_, cplx c_);

  /// Replace a matching right-hand side with its solution.
  void solve(std::vector<cplx>& d) const;
};

} // namespace num
