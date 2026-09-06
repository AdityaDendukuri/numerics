/// @file evidence.hpp
/// @brief Immutable evidence that a runtime value satisfies a law.
///
/// `claims<T, L>` (models.hpp) asks what a *type* asserts. This header is about what a
/// particular *value* has been shown to satisfy, and by what means: `assume<L>(x)` records
/// a caller's promise, `require<L>(x)` records an exhaustive check. Both carry an
/// `evidence_provenance` naming the origin and the source location, so a later failure can
/// be traced to the line that made the claim.
#pragma once

#include "core/math/associated.hpp"
#include "core/math/models.hpp"
#include <concepts>
#include <source_location>
#include <stdexcept>
#include <string_view>
#include <type_traits>

namespace num::math {

enum class evidence_origin { assumed, verified };

/// How and where a runtime value acquired mathematical evidence.
struct evidence_provenance final {
    evidence_origin origin;
    std::source_location location;
    std::string_view method;
};

/// @brief A law that can be attached to a runtime value as evidence.
///
/// Only operator laws qualify: evidence is about a specific object, and the space laws
/// describe a type. `law_tag` (models.hpp) is the broader membership in the hierarchy.
template <class P>
concept mathematical_proposition = std::derived_from<P, law::linear_map>;

template <class T, class P>
struct evidence_validator {
    static constexpr bool available = false;
};

namespace detail {
struct evidence_access;
}

template <class T, class... Properties>
class certified_ref final {
  public:
    using value_type = T;
    using math_laws = type_list<Properties...>;

    template <class... OtherProperties>
    requires(claims<certified_ref<T, OtherProperties...>, Properties> &&...) constexpr certified_ref(
        const certified_ref<T, OtherProperties...> &stronger) noexcept
        : value_(&stronger.get()), provenance_(stronger.provenance()) {}

    [[nodiscard]] constexpr const T &get() const noexcept { return *value_; }
    [[nodiscard]] constexpr const T &base() const noexcept { return *value_; }
    [[nodiscard]] constexpr const evidence_provenance &provenance() const noexcept {
        return provenance_;
    }

    [[nodiscard]] constexpr auto rows() const requires requires(const T &value) { value.rows(); }
    {
        return value_->rows();
    }

    [[nodiscard]] constexpr auto cols() const requires requires(const T &value) { value.cols(); }
    {
        return value_->cols();
    }

  private:
    friend struct detail::evidence_access;

    constexpr certified_ref(const T &value, evidence_provenance provenance) noexcept
        : value_(&value), provenance_(provenance) {}

    const T *value_;
    evidence_provenance provenance_;
};

namespace detail {

struct evidence_access final {
    template <class T, class... Properties>
    [[nodiscard]] static constexpr certified_ref<T, Properties...>
    make(const T &value, evidence_provenance provenance) noexcept {
        return certified_ref<T, Properties...>(value, provenance);
    }
};

} // namespace detail

template <class T, class... Ps>
struct claims_of<certified_ref<T, Ps...>> : claims_of<T> {};

namespace detail {

template <class T, class... Ps>
struct scalar_of<certified_ref<T, Ps...>, void> : scalar_of<T> {};

template <class T, class... Ps>
struct domain_of<certified_ref<T, Ps...>, void> : domain_of<T> {};

template <class T, class... Ps>
struct codomain_of<certified_ref<T, Ps...>, void> : codomain_of<T> {};

} // namespace detail

/// Attach caller-supplied evidence. Decidable shape prerequisites remain enforced.
template <class P, class T>
requires mathematical_proposition<P> [[nodiscard]] certified_ref<T, P>
assume(const T &value, std::source_location location = std::source_location::current()) {
    if constexpr (std::derived_from<P, law::endomorphism>) {
        if (value.rows() != value.cols()) {
            throw std::invalid_argument(
                "cannot certify a self-adjoint property on a non-square value");
        }
    }
    return detail::evidence_access::make<T, P>(
        value, {evidence_origin::assumed, location, "explicit caller assumption"});
}

// certified_ref is non-owning.  Binding it to a temporary would leave dangling
// evidence at the end of the full expression, so rvalues are rejected even
// though a const reference could otherwise bind to them.
template <class P, class T>
requires mathematical_proposition<P> certified_ref<T, P>
assume(const T &&, std::source_location = std::source_location::current()) = delete;

/// Exhaustively validate P using a type-specific validator before attaching evidence.
template <class P, class T>
requires mathematical_proposition<P> &&evidence_validator<T, P>::available
    [[nodiscard]] certified_ref<T, P>
    require(const T &value, std::source_location location = std::source_location::current()) {
    if (!evidence_validator<T, P>::verify(value)) {
        throw std::invalid_argument("value does not satisfy the required mathematical proposition");
    }
    return detail::evidence_access::make<T, P>(
        value, {evidence_origin::verified, location, "exhaustive validator"});
}

template <class P, class T>
requires mathematical_proposition<P> &&evidence_validator<T, P>::available certified_ref<T, P>
require(const T &&, std::source_location = std::source_location::current()) = delete;

} // namespace num::math

namespace num {
using math::assume;
using math::require;
} // namespace num
