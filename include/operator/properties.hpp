/// @file operator/properties.hpp
/// @brief Attaching a law to a matrix or an operator: `with_law`, `verify` and `assume`.
///
/// `assume_spd(A)` returns `with_law<decltype(A), law::spd>`, which claims SPD and so every
/// law SPD implies. Attaching a law samples it under `NUMERICS_DIAGNOSTICS`; see
/// `core/debug.hpp`. Squareness is decidable, so it is always checked.
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

/// @brief A matrix or operator `T` together with a law `L` it claims.
///
/// It owns `T`, and forwards the shape, the entries and the action. A matrix without an
/// `apply` acts through its entries.
template <class T, class L>
class with_law final {
  public:
    using laws = law::list<L>;
    using domain_type = typename detail::law_space<T>::type;
    using codomain_type = domain_type;

    /// Attach `L` without checking it. `num::assume` samples it first.
    explicit with_law(T value) : value_(std::move(value)) {}

    /// A value carrying a stronger law also carries `L`.
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

/// @brief Sample law `L` on `A`, and every law it implies.
///
/// The shape is always checked. The probes run under the active diagnostic level; each
/// rejects a violation, and none can prove the law holds.
/// @throws std::invalid_argument If `A` is not square.
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

/// @brief Attach law `L` to `A` after sampling it; see `verify`.
template <class L, class T>
[[nodiscard]] inline with_law<T, L> assume(T A, call_site site = {}) {
    with_law<T, L> claimed(std::move(A));
    verify<L>(claimed, site);
    return claimed;
}

/// @brief Attach \f$A = A^*\f$, which MINRES, Lanczos and `eig_sym` require.
template <class T>
[[nodiscard]] inline with_law<T, law::self_adjoint> assume_symmetric(T A, call_site site = {}) {
    return assume<law::self_adjoint>(std::move(A), site);
}

/// @brief Attach \f$\langle x, Ax \rangle \geq 0\f$.
template <class T>
[[nodiscard]] inline with_law<T, law::psd> assume_psd(T A, call_site site = {}) {
    return assume<law::psd>(std::move(A), site);
}

/// @brief Attach \f$\langle x, Ax \rangle > 0\f$, which CG, PCG and Cholesky require.
template <class T>
[[nodiscard]] inline with_law<T, law::spd> assume_spd(T A, call_site site = {}) {
    return assume<law::spd>(std::move(A), site);
}

/// @brief Attach strict diagonal dominance, which Jacobi, Gauss--Seidel and LU without
/// pivoting require. It is checked exactly, since the test reads the matrix once.
template <class T>
[[nodiscard]] inline with_law<T, law::diagonally_dominant>
assume_diagonally_dominant(T A, call_site site = {}) {
    return assume<law::diagonally_dominant>(std::move(A), site);
}

} // namespace num
