/// @file linalg/eigen/backends/omp/impl.hpp
/// @brief Private declarations for the OpenMP eigen backend.
#pragma once

#include "linalg/eigen/jacobi_eig.hpp"

namespace num::backends::omp {

EigenResult eig_sym(const Matrix& A, real tol, idx max_sweeps);

} // namespace num::backends::omp
