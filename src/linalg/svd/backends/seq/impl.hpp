/// @file linalg/svd/backends/seq/impl.hpp
/// @brief Private declarations for the sequential SVD backend.
#pragma once

#include "linalg/svd/svd.hpp"

namespace num::backends::seq {

SVDResult svd(const Matrix& A, real tol, idx max_sweeps);

} // namespace num::backends::seq
