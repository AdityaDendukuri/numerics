/// @file pde/poisson.hpp
/// @brief 2D Poisson equation solved via the Discrete Sine Transform.
///
/// Solves the homogeneous Dirichlet problem
/// \f[
///   -\Delta u(x,y) = f(x,y), \qquad (x,y) \in (0,1)^2,
///   \qquad u|_{\partial\Omega}=0 .
/// \f]
///
/// On an \f$N \times N\f$ interior grid with \f$h = 1/(N+1)\f$, the finite
/// difference variant diagonalizes
/// \f[
///   L_{2D} = L_{1D} \otimes I + I \otimes L_{1D}
/// \f]
/// with the DST-I basis
/// \f[
///   \phi_k(j) = \sin\!\left(\frac{j k \pi}{N+1}\right).
/// \f]
/// The transformed solve is
/// \f[
///   \hat{u}_{pq}
///     = \frac{h^2 \hat{f}_{pq}}
///            {2(1-\cos(p\pi/(N+1))) + 2(1-\cos(q\pi/(N+1)))} .
/// \f]
/// @todo Add Neumann/periodic Poisson variants, 3D tensor-product solves, and
/// Helmholtz shifted Poisson solves.
#pragma once

#include "container/matrix.hpp"

namespace num {
namespace pde {

/// Solve \f$-\Delta u=f\f$ using finite-difference eigenvalues.
[[nodiscard]] Matrix poisson2d_fd(const Matrix &f, int N);

/// Solve \f$-\Delta u=f\f$ using continuous eigenvalues \f$(k\pi)^2\f$.
[[nodiscard]] Matrix poisson2d(const Matrix &f, int N);

} // namespace pde
} // namespace num
