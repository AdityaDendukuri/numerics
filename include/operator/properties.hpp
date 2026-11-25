/// @file operator/properties.hpp
/// @brief Laws attached to operators: `with_law`, `verify`, `assume`.
#pragma once

#include "algebra/debug.hpp"
#include "algebra/scalar.hpp"
#include "container/vector.hpp"
#include "core/call_site.hpp"
#include "core/math/concepts.hpp"
#include "linear/math_adapters.hpp"
#include <concepts>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace num {

namespace detail {

template <class T>
struct law_space {
    using type = math::domain_t<T>;
};

template <class T>
requires std::is_void_v<math::domain_t<T>> && field<entry_t<T>>
struct law_space<T> {
    using type = vec<entry_t<T>>;
};

} // namespace detail

/// @brief Operator with a claimed law: `T` claiming `L`.
template <class T, class L>
class with_law final {
  public:
    using laws = law::list<L>;
    using domain_type = typename detail::law_space<T>::type;
    using codomain_type = domain_type;

    /// @brief Attach `L` unchecked; `assume` samples it first.
    explicit with_law(T value) : value_(std::move(value)) {}

    /// @brief Conversion from a stronger law.
    template <class Stronger>
    requires std::derived_from<Stronger, L> && (!std::same_as<Stronger, L>)
    with_law(with_law<T, Stronger> stronger) : value_(std::move(stronger.value_)) {}

    [[nodiscard]] const T &base() const noexcept { return value_; }
    [[nodiscard]] idx rows() const { return static_cast<idx>(value_.rows()); }
    [[nodiscard]] idx cols() const { return static_cast<idx>(value_.cols()); }

    [[nodiscard]] decltype(auto) operator()(idx i, idx j) const
    requires requires(const T &a) { a(i, j); }
    {
        return value_(i, j);
    }

    template <class X, class Y>
    void apply(const X &x, Y &y) const {
        if constexpr (requires { math::apply(value_, x, y); }) {
            math::apply(value_, x, y);
        } else {
            for (idx i = 0; i < rows(); ++i) {
                entry_t<T> sum{};
                for (idx j = 0; j < cols(); ++j) {
                    sum += value_(i, j) * x[j];
                }
                y[i] = sum;
            }
        }
    }

  private:
    template <class, class>
    friend class with_law;

    T value_;
};

/// @brief Sample law `L` and every law it implies on `A`.
/// @throws std::invalid_argument If `A` is not square.
/// @note Sampling can only reject a violation; it never proves the law holds.
template <class L, class T>
inline void verify(const T &A, call_site site = {}) {
    if (A.rows() != A.cols()) {
        throw std::invalid_argument("a matrix or operator claiming a law must be square");
    }
    using V = typename with_law<T, L>::domain_type;
    const auto loc = site.location;
    const idx n = A.cols();
    if constexpr (std::derived_from<L, law::diagonally_dominant>) {
        debug::verify_diagonal_dominance(A, loc);
    }
    if constexpr (std::derived_from<L, law::self_adjoint>) {
        debug::verify_linearity_sample<T, V>(A, n, loc);
        debug::verify_symmetry_sample<T, V>(A, n, loc);
    }
    if constexpr (std::derived_from<L, law::psd>) {
        debug::verify_psd_sample<T, V>(A, n, loc);
    }
    if constexpr (std::derived_from<L, law::spd>) {
        debug::verify_spd_sample<T, V>(A, n, loc);
    }
}

/// @brief Attach law `L` after sampling it.
template <class L, class T>
[[nodiscard]] inline with_law<T, L> assume(T A, call_site site = {}) {
    with_law<T, L> claimed(std::move(A));
    verify<L>(claimed, site);
    return claimed;
}

/// @brief Attach self-adjointness: \f$A = A^*\f$.
template <class T>
[[nodiscard]] inline with_law<T, law::self_adjoint> assume_symmetric(T A, call_site site = {}) {
    return assume<law::self_adjoint>(std::move(A), site);
}

/// @brief Attach positive semidefiniteness: \f$\langle x, A x \rangle \ge 0\f$.
template <class T>
[[nodiscard]] inline with_law<T, law::psd> assume_psd(T A, call_site site = {}) {
    return assume<law::psd>(std::move(A), site);
}

/// @brief Attach positive definiteness: \f$\langle x, A x \rangle > 0\f$.
template <class T>
[[nodiscard]] inline with_law<T, law::spd> assume_spd(T A, call_site site = {}) {
    return assume<law::spd>(std::move(A), site);
}

/// @brief Attach diagonal dominance: \f$|a_{ii}| > \sum_{j \ne i} |a_{ij}|\f$. Checked exactly.
template <class T>
[[nodiscard]] inline with_law<T, law::diagonally_dominant>
assume_diagonally_dominant(T A, call_site site = {}) {
    return assume<law::diagonally_dominant>(std::move(A), site);
}

} // namespace num
