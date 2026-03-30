/// @file factorization/thomas.hpp
/// @brief Thomas algorithm  -- direct O(n) tridiagonal solver
///
/// Solves the tridiagonal system:
///
///   [ b[0]  c[0]                     ] [ x[0]   ]   [ d[0]   ]
///   [ a[0]  b[1]  c[1]               ] [ x[1]   ] = [ d[1]   ]
///   [       a[1]  b[2]  c[2]         ] [ x[2]   ]   [ d[2]   ]
///   [          ...   ...   ...       ] [   :    ]   [   :    ]
///   [             a[n-2] b[n-1]      ] [ x[n-1] ]   [ d[n-1] ]
///
/// Algorithm: LU factorisation of the tridiagonal structure, O(n) time and
/// O(n) extra space.  Numerically stable for strictly diagonally dominant
/// or symmetric positive definite tridiagonals; may fail for singular pivots.
#pragma once

#include "core/types.hpp"
#include "core/vector.hpp"
#include "core/policy.hpp"

namespace num {

/// @brief Thomas algorithm (LU for tridiagonal systems), O(n).
///
/// @param a  Sub-diagonal, size n-1
/// @param b  Main diagonal, size n
/// @param c  Super-diagonal, size n-1
/// @param d  Right-hand side vector, size n
/// @param x  Solution vector (output), size n
/// @param backend  Backend::lapack uses LAPACKE_dgtsv (default when available).
///                 Backend::seq    uses our 3-sweep O(n) implementation.
///                 Backend::gpu    uses CUDA batched tridiagonal (batch=1).
void thomas(const Vector& a, const Vector& b, const Vector& c,
            const Vector& d, Vector& x, Backend backend = lapack_backend);

} // namespace num
