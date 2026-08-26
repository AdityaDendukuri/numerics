/// @file pde/pde.hpp
/// @brief Umbrella include for the PDE module.
/// @todo Add boundary-condition objects, Helmholtz operators, variable-
/// coefficient elliptic operators, and structured 1D/2D/3D operator builders.
#pragma once

#include "pde/grid_operators.hpp"

#include "pde/adi.hpp"
#include "pde/diffusion.hpp"
#include "pde/field_solver.hpp"
#include "pde/poisson.hpp"
#include "pde/stencil.hpp"
