/// @file ode/concepts.hpp
/// @brief Contracts for initial value problems and the state spaces they evolve in.
#pragma once

#include "container/concepts.hpp"
#include "ode/types.hpp"
#include <concepts>

namespace num {

/// @brief State exposing its underlying vector space to implicit integrators.
///
/// Implicit steppers solve a linear system in the state space, so they need the
/// state as a vector rather than as a field, grid or particle set. The space is a
/// parameter refining `vector_space`, not the concrete `vec<real>`: a state over complex
/// amplitudes or single precision exposes its own space just as well, and the earlier
/// form — which named `vec<real>` outright — excluded both.
template <class T, class V = vec<real>>
concept vec_field = vector_space<V> && requires(T &field) {
    { field.as_vec() } -> std::same_as<V &>;
};

} // namespace num
