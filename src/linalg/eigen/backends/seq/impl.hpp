/// @file linalg/eigen/backends/seq/impl.hpp
/// @brief Private declarations for the sequential eigen backend.
#pragma once

#include "linalg/eigen/jacobi_eig.hpp"

namespace num::backends::seq {

EigenResult eig_sym(const Matrix& A, real tol, idx max_sweeps);

} // namespace num::backends::seq
