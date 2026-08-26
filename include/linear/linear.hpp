/// @file linear/linear.hpp
/// @brief Umbrella include for linear algebra routines.
#pragma once

#include "linear/banded/banded.hpp"
#include "linear/concepts.hpp"
#include "linear/debug.hpp"
#include "linear/eigen/eigen.hpp"
#include "linear/expv/expv.hpp"
#include "linear/factorization/factorization.hpp"
#include "linear/math_adapters.hpp"
#include "linear/matrix_properties.hpp"
#include "linear/matrix_utils.hpp"
#include "linear/subspace.hpp"
#include "linear/solvers/dense_resolvent.hpp"
#include "linear/solvers/solvers.hpp"
#include "linear/solvers/sparse_resolvent.hpp"
#include "linear/sparse/klu.hpp"
#include "linear/sparse/sparse.hpp"
#include "linear/sparse/umfpack.hpp"
#include "linear/svd/svd.hpp"
