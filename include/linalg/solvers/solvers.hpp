/// @file solvers.hpp
/// @brief Umbrella include for all linear solvers
///
/// Including this header provides every solver in the library.
/// Individual headers (cg.hpp, gauss_seidel.hpp, jacobi.hpp, krylov.hpp)
/// can also be included directly.
#pragma once

#include "linalg/solvers/cg.hpp"
#include "linalg/solvers/gauss_seidel.hpp"
#include "linalg/solvers/jacobi.hpp"
#include "linalg/solvers/krylov.hpp"
#include "linalg/solvers/solver_result.hpp"
