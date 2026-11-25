/// @file pde/poisson.hpp
/// @brief 2D Poisson solves by discrete sine transform: \f$-\Delta u = f\f$, zero Dirichlet.
// TODO: Neumann and periodic variants, 3D solves, shifted (Helmholtz) solves.
#pragma once

#include "container/matrix.hpp"

namespace num {
namespace pde {

/// @brief Poisson solve, finite-difference eigenvalues: \f$-\Delta_h u = f\f$.
[[nodiscard]] mat<real> poisson2d_fd(const mat<real> &f, int N);

/// @brief Poisson solve, continuous eigenvalues \f$(k\pi)^2\f$: \f$-\Delta u = f\f$.
[[nodiscard]] mat<real> poisson2d(const mat<real> &f, int N);

} // namespace pde
} // namespace num
