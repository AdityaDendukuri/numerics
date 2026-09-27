/// @file operations.hpp
/// @brief The vector space operations, shared by concepts, algorithms, and tests.
///
/// Each operation calls a `tag_invoke` overload when the type provides one, and otherwise
/// loops over `v[i]`. Each is constrained, so it is not callable on a type it cannot handle.
#pragma once

#include "core/math/associated.hpp"
#include <cmath>
#include <complex>
#include <concepts>
#include <cstddef>
#include <type_traits>
#include <utility>

namespace num::math {

namespace cpo_detail {

void tag_invoke();

template <class Tag, class... Args>
concept tag_invocable = requires(Tag tag, Args &&...args) {
    tag_invoke(tag, std::forward<Args>(args)...);
};

/// A sized container of scalars, written entry by entry.
template <class V>
concept indexable = requires(const V &cv, V &v, std::size_t i) {
    { cv.size() } -> std::integral;
    cv[i];
    v[i] = cv[i];
};

template <class T>
[[nodiscard]] constexpr auto conjugate(const T &x) {
    if constexpr (scalars::is_complex_v<T>) {
        return std::conj(x);
    } else {
        return x;
    }
}

template <class T>
[[nodiscard]] constexpr auto real_part(const T &x) {
    if constexpr (scalars::is_complex_v<T>) {
        return x.real();
    } else {
        return x;
    }
}

} // namespace cpo_detail

struct dimension_t {
    template <class V>
    requires cpo_detail::tag_invocable<dimension_t, const V &> || cpo_detail::indexable<V>
    [[nodiscard]] constexpr auto operator()(const V &v) const {
        if constexpr (cpo_detail::tag_invocable<dimension_t, const V &>) {
            return tag_invoke(*this, v);
        } else {
            return v.size();
        }
    }
};

inline constexpr dimension_t dimension{};

struct zero_like_t {
    template <class V>
    requires cpo_detail::tag_invocable<zero_like_t, const V &> ||
             (cpo_detail::indexable<V> && std::constructible_from<V, std::size_t>)
    [[nodiscard]] auto operator()(const V &exemplar) const {
        if constexpr (cpo_detail::tag_invocable<zero_like_t, const V &>) {
            return tag_invoke(*this, exemplar);
        } else {
            return V(dimension(exemplar));
        }
    }
};

struct scale_t {
    template <class A, class V>
    requires cpo_detail::tag_invocable<scale_t, A, V &> || cpo_detail::indexable<V>
    void operator()(A a, V &v) const {
        if constexpr (cpo_detail::tag_invocable<scale_t, A, V &>) {
            tag_invoke(*this, a, v);
        } else {
            for (decltype(dimension(v)) i = 0; i < dimension(v); ++i) {
                v[i] = a * v[i];
            }
        }
    }
};

struct axpy_t {
    template <class A, class X, class Y>
    requires cpo_detail::tag_invocable<axpy_t, A, const X &, Y &> ||
             (cpo_detail::indexable<X> && cpo_detail::indexable<Y>)
    void operator()(A a, const X &x, Y &y) const {
        if constexpr (cpo_detail::tag_invocable<axpy_t, A, const X &, Y &>) {
            tag_invoke(*this, a, x, y);
        } else {
            for (decltype(dimension(y)) i = 0; i < dimension(y); ++i) {
                y[i] = y[i] + (a * x[i]);
            }
        }
    }
};

struct linear_combination_t {
    template <class A, class X, class B, class Y>
    requires cpo_detail::tag_invocable<linear_combination_t, A, const X &, B, Y &> ||
             (cpo_detail::indexable<X> && cpo_detail::indexable<Y>)
    void operator()(A a, const X &x, B b, Y &y) const {
        if constexpr (cpo_detail::tag_invocable<linear_combination_t, A, const X &, B, Y &>) {
            tag_invoke(*this, a, x, b, y);
        } else {
            for (decltype(dimension(y)) i = 0; i < dimension(y); ++i) {
                y[i] = (a * x[i]) + (b * y[i]);
            }
        }
    }
};

struct inner_t {
    template <class X, class Y>
    requires cpo_detail::tag_invocable<inner_t, const X &, const Y &> ||
             (cpo_detail::indexable<X> && cpo_detail::indexable<Y>)
    [[nodiscard]] auto operator()(const X &x, const Y &y) const {
        if constexpr (cpo_detail::tag_invocable<inner_t, const X &, const Y &>) {
            return tag_invoke(*this, x, y);
        } else {
            scalar_t<X> result{};
            for (decltype(dimension(x)) i = 0; i < dimension(x); ++i) {
                result += cpo_detail::conjugate(x[i]) * y[i];
            }
            return result;
        }
    }
};

inline constexpr inner_t inner{};
inline constexpr axpy_t axpy{};

struct axpy_norm_sq_t {
    template <class A, class X, class Y>
    requires cpo_detail::tag_invocable<axpy_norm_sq_t, A, const X &, Y &> ||
             (cpo_detail::indexable<X> && cpo_detail::indexable<Y>)
    [[nodiscard]] auto operator()(A a, const X &x, Y &y) const {
        if constexpr (cpo_detail::tag_invocable<axpy_norm_sq_t, A, const X &, Y &>) {
            return tag_invoke(*this, a, x, y);
        } else {
            axpy(a, x, y);
            return cpo_detail::real_part(inner(y, y));
        }
    }
};

struct norm_t {
    template <class V>
    requires cpo_detail::tag_invocable<norm_t, const V &> || cpo_detail::indexable<V>
    [[nodiscard]] auto operator()(const V &v) const {
        if constexpr (cpo_detail::tag_invocable<norm_t, const V &>) {
            return tag_invoke(*this, v);
        } else {
            return std::sqrt(cpo_detail::real_part(inner(v, v)));
        }
    }
};

struct apply_t {
    template <class Op, class X, class Y>
    requires cpo_detail::tag_invocable<apply_t, const Op &, const X &, Y &> ||
             requires(const Op &op, const X &x, Y &y) { op.apply(x, y); }
    void operator()(const Op &op, const X &x, Y &y) const {
        if constexpr (cpo_detail::tag_invocable<apply_t, const Op &, const X &, Y &>) {
            tag_invoke(*this, op, x, y);
        } else {
            op.apply(x, y);
        }
    }
};

inline constexpr zero_like_t zero_like{};
inline constexpr scale_t scale{};
inline constexpr linear_combination_t linear_combination{};
inline constexpr axpy_norm_sq_t axpy_norm_sq{};
inline constexpr norm_t norm{};
inline constexpr apply_t apply{};

} // namespace num::math
