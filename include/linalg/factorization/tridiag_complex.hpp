/// @file factorization/tridiag_complex.hpp
/// @brief Precomputed LU Thomas solver for constant-coefficient complex
/// tridiagonal systems.
///
/// Solves the n x n system:
///
///   a*x[k-1] + b*x[k] + c*x[k+1] = d[k],  k = 0 .. n-1
///
/// with Dirichlet boundary conditions  x[-1] = x[n] = 0.
///
/// Because a, b, c are the same for every row the LU factorization depends only
/// on the coefficients, not on d.  Factor once, then call solve() for each
/// new right-hand side in O(n).
///
/// Typical use: Crank-Nicolson kinetic sweeps in the TDSE solver, where
/// a = c = -i*alpha  and  b = 1 + 2i*alpha  for some real alpha.
#pragma once

#include "core/types.hpp"
#include <complex>
#include <vector>

namespace num {

/// @brief Constant-coefficient complex tridiagonal solver (precomputed LU).
///
/// @code
/// num::ComplexTriDiag td;
/// std::complex<double> a(0.0, -alpha);
/// std::complex<double> b(1.0,  2.0*alpha);
/// std::complex<double> c(0.0, -alpha);
/// td.factor(n, a, b, c);
///
/// std::vector<std::complex<double>> rhs = ...;
/// td.solve(rhs);   // rhs is overwritten with the solution
/// @endcode
struct ComplexTriDiag {
    using cplx = std::complex<double>;

    std::vector<cplx> c_mod; ///< Modified super-diagonal (precomputed from LU)
    std::vector<cplx>
         inv_b;        ///< Inverse of modified main diagonal (precomputed)
    int  n       = 0;
    cplx a_coeff = {}; ///< Sub-diagonal value (constant across all rows)

    /// @brief Factor the tridiagonal matrix.
    ///
    /// @param n_  System size
    /// @param a_  Sub-diagonal coefficient (row k depends on x[k-1])
    /// @param b_  Main-diagonal coefficient
    /// @param c_  Super-diagonal coefficient (row k depends on x[k+1])
    void factor(int n_, cplx a_, cplx b_, cplx c_);

    /// @brief In-place Thomas solve.
    ///
    /// On entry  d holds the right-hand side; on exit it holds the solution.
    /// The size of d must equal n (set by the last factor() call).
    void solve(std::vector<cplx>& d) const;
};

} // namespace num
