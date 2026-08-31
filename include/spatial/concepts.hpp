/// @file spatial/concepts.hpp
/// @brief Contracts for spatial acceleration structures and smoothing kernels.
///
/// Coordinates live in a scalar field, so the structures here are stated over
/// `num::Field`. An integer coordinate type would place particles in the wrong
/// cells and report nothing, and the constraint rejects it.
#pragma once

#include "algebra/concepts.hpp"
#include "core/types.hpp"
#include <concepts>
#include <utility>

namespace num {

/// @brief Callable returning the position of particle \f$i\f$ as \f$(x, y)\f$.
///
/// The acceleration structures never own coordinates. A caller supplies an
/// accessor, so positions may live in a struct of arrays, an array of structs,
/// or a simulation's own particle type.
template <class A, class Scalar = real>
concept PositionAccessor2D = scalars::Field<Scalar> && requires(const A &get_pos, int i) {
    { get_pos(i) } -> std::convertible_to<std::pair<Scalar, Scalar>>;
};

/// @brief Structure answering which particles lie near a point.
///
/// The contract is a visitor rather than a returned container: `query` calls
/// `f(j)` for each candidate, so nothing allocates in the inner loop.
template <class L, class Scalar = real>
concept NeighborQuery2D = scalars::Field<Scalar> &&
    requires(const L &list, Scalar px, Scalar py) {
    list.query(px, py, [](int) {});
};

/// @brief Radially symmetric smoothing kernel \f$W(r, h)\f$ with its radial derivative.
///
/// An SPH kernel is normalized so that \f$\int W \, dV = 1\f$ over its support and
/// vanishes beyond \f$r = 2h\f$. Both are properties of the values rather than the
/// type, and `num::spatial::debug::verify_kernel_normalization` samples them.
template <class K, class Scalar = float>
concept SmoothingKernel = scalars::Field<Scalar> && requires(Scalar r, Scalar h) {
    { K::W(r, h) } -> std::convertible_to<Scalar>;
    { K::dW_dr(r, h) } -> std::convertible_to<Scalar>;
};

/// @brief Site-indexed lattice supplying periodic nearest neighbours.
///
/// Each site \f$i\f$ on an \f$N \times N\f$ lattice has four neighbours obtained by
/// stepping one row or column with wraparound. The tables are precomputed so a
/// sweep never evaluates a modulus.
template <class P>
concept PeriodicLattice2D = requires(const P &lattice, int i) {
    { lattice.N } -> std::convertible_to<int>;
    { lattice.up[i] } -> std::convertible_to<int>;
    { lattice.dn[i] } -> std::convertible_to<int>;
    { lattice.lt[i] } -> std::convertible_to<int>;
    { lattice.rt[i] } -> std::convertible_to<int>;
};

} // namespace num
