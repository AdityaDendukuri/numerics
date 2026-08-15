/// @file factorization/thomas.hpp
/// @brief Thomas algorithm for tridiagonal systems.
///
/// Solves \f$a_{i-1}x_{i-1}+b_i x_i+c_i x_{i+1}=d_i\f$ in \f$O(n)\f$.
#pragma once

#include "core/policy.hpp"
#include "core/types.hpp"
#include "core/vector.hpp"

namespace num {

void thomas(const Vector& a,
            const Vector& b,
            const Vector& c,
            const Vector& d,
            Vector& x,
            Backend backend = lapack_backend);

} // namespace num
