/// @file linalg/eigen/backends/lapack/impl.hpp
/// @brief Private declarations for the LAPACK eigen backend.
#pragma once

#include "linalg/eigen/jacobi_eig.hpp"

namespace num::backends::lapack {

EigenResult eig_sym(const Matrix& A);

} // namespace num::backends::lapack
