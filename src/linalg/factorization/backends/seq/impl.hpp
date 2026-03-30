/// @file linalg/factorization/backends/seq/impl.hpp
/// @brief Private declarations for the sequential factorization backend.
/// Only included by the dispatcher .cpp files in src/linalg/factorization/.
#pragma once

#include "linalg/factorization/lu.hpp"
#include "linalg/factorization/qr.hpp"
#include "linalg/factorization/thomas.hpp"

namespace num::backends::seq {

LUResult lu(const Matrix& A);
QRResult qr(const Matrix& A);
void     thomas(const Vector& a, const Vector& b, const Vector& c,
                const Vector& d, Vector& x);

} // namespace num::backends::seq
