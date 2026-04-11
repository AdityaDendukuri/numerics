/// @file pde/pde.hpp
/// @brief Umbrella include for the PDE module.
///
/// Finite-difference operators, 3D field types, and time integration:
///   stencil.hpp   -- 2D/3D Laplacians, fiber sweeps, gradient, divergence,
///   curl fields.hpp    -- ScalarField3D, VectorField3D, FieldSolver,
///   MagneticSolver adi.hpp       -- CrankNicolsonADI (2D parabolic, Strang
///   splitting) diffusion.hpp -- diffusion_step_2d{,_dirichlet} (explicit
///   Euler)
#pragma once

#include "pde/stencil.hpp"
#include "pde/fields.hpp"
#include "pde/adi.hpp"
#include "pde/diffusion.hpp"
#include "pde/poisson.hpp"
