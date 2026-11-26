/// @file lapack/lapack_wrapper.hpp
/// @brief LAPACKE, when the build has LAPACK.
#pragma once

#if defined(NUMERICS_HAS_LAPACK)
#include <lapacke.h>
#endif
