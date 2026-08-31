/// @file operator/operator.hpp
/// @brief Umbrella for generic operator adapters.
///
/// The LinearOperator contract itself lives in operator/concepts.hpp. The
/// SparseMatrix adapter lives in linear/sparse/sparse_op.hpp (include it
/// directly) so this module stays free of any linear-algebra dependency.
#pragma once

#include "algebra/properties.hpp"
#include "operator/callable.hpp"
#include "operator/concepts.hpp"
#include "operator/dense.hpp"
#include "operator/projected.hpp"
#include "operator/properties.hpp"
