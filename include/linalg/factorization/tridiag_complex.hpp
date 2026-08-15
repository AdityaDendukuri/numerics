/// @file factorization/tridiag_complex.hpp
/// @brief Precomputed Thomas solver for constant-coefficient complex systems.
///
/// Solves \f$a x_{k-1}+b x_k+c x_{k+1}=d_k\f$ with fixed \f$a,b,c\f$.
#pragma once

#include "core/types.hpp"
#include <complex>
#include <vector>

namespace num {

struct ComplexTriDiag {
  using cplx = std::complex<double>;

  std::vector<cplx> c_mod;
  std::vector<cplx> inv_b;
  int n = 0;
  cplx a_coeff = {};

  void factor(int n_, cplx a_, cplx b_, cplx c_);

  void solve(std::vector<cplx>& d) const;
};

} // namespace num
