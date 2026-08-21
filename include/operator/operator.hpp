/// @file operator/operator.hpp
/// @brief Umbrella for generic operator adapters.
///
/// The LinearOperator contract itself lives in operator/concepts.hpp. The
/// SparseMatrix adapter lives in linalg/sparse/sparse_op.hpp (include it
/// directly) so this module stays free of any linalg dependency.
#pragma once

#include "operator/callable.hpp"
#include "operator/concepts.hpp"
#include "operator/dense.hpp"
#include "operator/properties.hpp"
