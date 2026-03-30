/// @file linalg/svd/backends/lapack/impl.hpp
/// @brief Private declarations for the LAPACK SVD backend.
#pragma once

#include "linalg/svd/svd.hpp"

namespace num::backends::lapack {

SVDResult svd(const Matrix& A);

} // namespace num::backends::lapack
