/// @file analysis/types.hpp
/// @brief Common function type aliases for the analysis module
#pragma once

#include "core/types.hpp"
#include <functional>

namespace num {

/// @brief Real-valued scalar function f(x)
using ScalarFn = std::function<real(real)>;

/// @brief Real-valued multivariate function f(x) where x is a scalar parameter
/// and the function returns a vector of residuals  -- used in nonlinear systems
using VectorFn = std::function<void(real, real *, real *)>;

} // namespace num
