/// @file algebra/ops.hpp
/// @brief vec space operations, resolved to whichever form a type provides.
///
/// A concept requiring `size()` and `operator[]` describes a *coordinate array*.
/// It does not describe a vector space, because it says nothing about closure —
/// whether a sum of two elements is itself an element, or whether a zero exists.
/// `std::span<const T>` is indexable over a field and is emphatically not a vector
/// space: there is nowhere to put the sum.
///
/// So the structure here is stated in two halves, matching the split in
/// `algebra/properties.hpp`:
///
///   - **Structure**, decided by the compiler: the space supplies the operations
///     (addition, scalar multiplication, a zero) and is closed under them.
///   - **Laws**, asserted and sampled: associativity, commutativity, distributivity,
///     conjugate symmetry of the inner product, the triangle inequality. No type
///     can prove these; `num::algebra::verify_*` probes them on real instances.
///
/// The samplers deliberately exercise *the type's own* operations where it has
/// them — `num::dot`, `num::norm`, `num::axpy` are found by argument-dependent
/// lookup — so the laws are checked against the shipped implementation, backend
/// dispatch included, rather than against a reimplementation written here.
#pragma once


#include "algebra/scalar.hpp"
#include "kernel/kernel.hpp"
#include "core/types.hpp"
#include <concepts>
#include <type_traits>
#include <utility>

namespace num::algebra {

// =============================================================================
// 1. Operation access
// =============================================================================
//
// Three presentations of the same algebra coexist in this library: free functions
// (num::vec), operators (num::small_vec), and neither (std::vector). These
// adaptors resolve to whichever a type provides, so the concepts and law samplers
// below are written once.

namespace detail {

using std::declval;

/// True when V exposes contiguous storage over a real field, so a raw kernel applies.
template <class V>
concept raw_reducible = std::floating_point<scalar_t<V>> && requires(const V &v) {
    { v.data() } -> std::convertible_to<const scalar_t<V> *>;
    { v.size() };
};

template <class V>
concept has_free_dot = requires(const V &x, const V &y) {
    { dot(x, y) };
};

template <class V>
concept has_free_norm = requires(const V &x) {
    { norm(x) };
};

template <class V, class T>
concept has_free_axpy = requires(T a, const V &x, V &y) {
    axpy(a, x, y);
};

template <class V, class T>
concept has_free_scale = requires(V &v, T a) {
    scale(v, a);
};

} // namespace detail

/// @brief Inner product \f$\langle x,y \rangle = \sum_i \overline{x_i} y_i\f$.
///
/// Uses the type's own `dot` when it has one, so what gets law-checked is the
/// shipped routine rather than a local reimplementation.
template <class V>
[[nodiscard]] inline scalar_t<V> inner(const V &x, const V &y) {
    using T = scalar_t<V>;
    if constexpr (detail::has_free_dot<V>) {
        return static_cast<T>(dot(x, y));
    } else if constexpr (detail::raw_reducible<V>) {
        // contiguous real storage: defer to the kernel rather than restating the loop.
        return kernel::dot(x.data(), y.data(), x.size());
    } else {
        T sum = T(0);
        for (idx i = 0; i < x.size(); ++i) {
            sum += scalars::conj(x[i]) * y[i];
        }
        return sum;
    }
}

/// @brief Norm \f$\|x\|\f$, taken from the type's own `norm` when available.
template <class V>
[[nodiscard]] inline scalars::real_t<scalar_t<V>> norm_of(const V &x) {
    if constexpr (detail::has_free_norm<V>) {
        return static_cast<scalars::real_t<scalar_t<V>>>(norm(x));
    } else if constexpr (detail::raw_reducible<V>) {
        return kernel::norm(x.data(), x.size());
    } else {
        return std::sqrt(scalars::re(inner(x, x)));
    }
}

/// @brief In-place update \f$y \leftarrow y + a x\f$.
template <class V>
inline void axpy_into(scalar_t<V> a, const V &x, V &y) {
    if constexpr (detail::has_free_axpy<V, scalar_t<V>>) {
        axpy(a, x, y);
    } else {
        for (idx i = 0; i < y.size(); ++i) {
            y[i] = y[i] + (a * x[i]);
        }
    }
}

/// @brief In-place scaling \f$v \leftarrow a v\f$.
template <class V>
inline void scale_inplace(V &v, scalar_t<V> a) {
    if constexpr (detail::has_free_scale<V, scalar_t<V>>) {
        scale(v, a);
    } else {
        for (idx i = 0; i < v.size(); ++i) {
            v[i] = a * v[i];
        }
    }
}

/// @brief The zero vector of dimension n; the additive identity.
template <class V>
[[nodiscard]] inline V zero(idx n) {
    V z(n);
    for (idx i = 0; i < n; ++i) {
        z[i] = scalar_t<V>(0);
    }
    return z;
}

} // namespace num::algebra
