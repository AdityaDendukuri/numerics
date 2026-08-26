/// @file subspace.hpp
/// @brief Runtime predicates representing mathematical linear subspaces.
#pragma once

#include "core/math/concepts.hpp"
#include <algorithm>
#include <cmath>
#include <concepts>
#include <type_traits>

namespace num::math {

struct contains_t {
    template <class Subspace, class V>
    [[nodiscard]] bool operator()(const Subspace &subspace, const V &value) const {
        if constexpr (cpo_detail::TagInvocable<contains_t, const Subspace &, const V &>) {
            return tag_invoke(*this, subspace, value);
        } else {
            return subspace.contains(value);
        }
    }
};

struct project_t {
    template <class Subspace, class V>
    void operator()(const Subspace &subspace, V &value) const {
        if constexpr (cpo_detail::TagInvocable<project_t, const Subspace &, V &>) {
            tag_invoke(*this, subspace, value);
        } else {
            subspace.project(value);
        }
    }
};

inline constexpr contains_t contains{};
inline constexpr project_t project{};

template <class Subspace, class V>
concept LinearSubspaceOf = Models<Subspace, law::linear_subspace> && VectorSpace<V> &&
                           requires(const Subspace &subspace, const V &constant, V &value) {
    {contains(subspace, constant)}->std::same_as<bool>;
    project(subspace, value);
};

} // namespace num::math

namespace num::space {

/// The codimension-one subspace {x : sum_i x_i = 0}.
struct zero_sum final {
    double tolerance = 1e-11;

    template <class V>
    requires std::floating_point<math::scalar_t<V>> &&requires(const V &value, std::size_t i) {
        value[i];
    }
    [[nodiscard]] bool contains(const V &value) const {
        using S = math::scalar_t<V>;
        S total = S(0);
        S magnitude = S(0);
        const auto n = math::dimension(value);
        using I = std::remove_cvref_t<decltype(n)>;
        for (I i = 0; i < n; ++i) {
            total += value[i];
            magnitude += std::abs(value[i]);
        }
        return std::abs(total) <= tolerance * std::max(S(1), magnitude);
    }

    template <class V>
    requires std::floating_point<math::scalar_t<V>> &&requires(V &value, std::size_t i) {
        value[i] = math::scalar_t<V>(0);
    }
    void project(V &value) const {
        using S = math::scalar_t<V>;
        const auto n = math::dimension(value);
        if (n == 0) {
            return;
        }
        S total = S(0);
        using I = std::remove_cvref_t<decltype(n)>;
        for (I i = 0; i < n; ++i) {
            total += value[i];
        }
        const S mean = total / static_cast<S>(n);
        for (I i = 0; i < n; ++i) {
            value[i] -= mean;
        }
    }
};

} // namespace num::space

namespace num::math {

template <>
struct model_of<space::zero_sum> {
    using laws = type_list<law::linear_subspace>;
};

} // namespace num::math
