/// @file spectral/concepts.hpp
/// @brief Contracts for discrete transforms.
#pragma once

#include "algebra/concepts.hpp"
#include "container/vector.hpp"
#include "core/types.hpp"
#include <concepts>

namespace num {

/// @brief Reusable plan executing a fixed-length transform.
///
/// \f[ X_k = \sum_{n=0}^{N-1} x_n \, e^{-2\pi i k n / N} \f]
///
/// A plan is built once for a length and reused, which is what lets a backend
/// precompute twiddle factors or hand the length to FFTW. Transform length is
/// fixed at construction, so a plan and a vector must agree on it.
template <class P, class V = cvec>
concept transform_plan = inner_product_space<V> && requires(const P &plan, const V &in, V &out) {
    { plan.size() } -> std::convertible_to<int>;
    plan.execute(in, out);
};

/// @brief Transform preserving the inner product up to a scale factor.
///
/// \f[ \sum_n |x_n|^2 = \frac{1}{N} \sum_k |X_k|^2 \f]
///
/// Parseval's identity is a statement about the inner product, which is why the
/// domain is required to be an inner product space rather than merely indexable.
/// It is the property that makes a spectral method conserve energy, and
/// `num::spectral::debug::verify_parseval` samples it.
template <class P, class V = cvec>
concept unitary_transform = transform_plan<P, V>;

} // namespace num
