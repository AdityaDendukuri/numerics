/// @file operator/operator.hpp
/// @brief Umbrella for generic operator adapters.
///
/// The LinearOperator contract itself lives in core/concepts.hpp. The
/// SparseMatrix adapter lives in linalg/sparse/sparse_op.hpp (include it
/// directly) so this module stays free of any linalg dependency.
#pragma once

#include "core/concepts.hpp"
#include "operator/callable.hpp"
#include "operator/dense.hpp"
#include "operator/properties.hpp"
