/// @file operator/operator.hpp
/// @brief Adapters that make matrices and functions into operators.
///
/// The linear_operator contract itself lives in operator/concepts.hpp. The
/// spmat adapter lives in linear/sparse/sparse_op.hpp (include it
/// directly) so this module stays free of any linear-algebra dependency.
#pragma once

#include "operator/callable.hpp"
#include "operator/concepts.hpp"
#include "operator/dense.hpp"
#include "operator/projected.hpp"
#include "operator/properties.hpp"
