/// @file evidence.hpp
/// @brief Immutable evidence for mathematical propositions about runtime values.
#pragma once

#include "core/math/associated.hpp"
#include "core/math/models.hpp"
#include <concepts>
#include <source_location>
#include <stdexcept>
#include <string_view>
#include <type_traits>

namespace num::axiom {

struct linear {};
struct square_operator : linear {};
struct self_adjoint : square_operator {};
struct positive_semidefinite : self_adjoint {};
struct positive_definite : positive_semidefinite {};

/// The restriction of an operator to Subspace is self-adjoint and maps the
/// subspace into itself.
template <class Subspace>
struct self_adjoint_on : square_operator {};

/// The restriction of an operator to Subspace is positive semidefinite.
template <class Subspace>
struct positive_semidefinite_on : self_adjoint_on<Subspace> {};

/// The restriction of an operator to Subspace is positive definite.
template <class Subspace>
struct positive_definite_on : positive_semidefinite_on<Subspace> {};

} // namespace num::axiom

namespace num::math {

enum class evidence_origin { assumed, verified };

/// How and where a runtime value acquired mathematical evidence.
struct EvidenceProvenance final {
    evidence_origin origin;
    std::source_location location;
    std::string_view method;
};

template <class T>
struct intrinsic_propositions {
    using type = type_list<>;
};

namespace detail {

template <class T, class = void>
struct declared_propositions {
    using type = typename intrinsic_propositions<std::remove_cvref_t<T>>::type;
};

template <class T>
struct declared_propositions<T, std::void_t<typename std::remove_cvref_t<T>::math_propositions>> {
    using type = typename std::remove_cvref_t<T>::math_propositions;
};

template <class P, class... Ps>
consteval bool carries_in(type_list<Ps...>) {
    return (std::derived_from<Ps, P> || ...);
}

} // namespace detail

template <class T, class P>
concept Carries = detail::carries_in<P>(typename detail::declared_propositions<T>::type{});

template <class P>
concept MathematicalProposition = std::derived_from<P, axiom::linear>;

template <class T, class P>
struct evidence_validator {
    static constexpr bool available = false;
};

namespace detail {
struct evidence_access;
}

template <class T, class... Properties>
class CertifiedRef final {
  public:
    using value_type = T;
    using math_propositions = type_list<Properties...>;

    template <class... OtherProperties>
    requires(Carries<CertifiedRef<T, OtherProperties...>, Properties> &&...) constexpr CertifiedRef(
        const CertifiedRef<T, OtherProperties...> &stronger) noexcept
        : value_(&stronger.get()), provenance_(stronger.provenance()) {}

    [[nodiscard]] constexpr const T &get() const noexcept { return *value_; }
    [[nodiscard]] constexpr const T &base() const noexcept { return *value_; }
    [[nodiscard]] constexpr const EvidenceProvenance &provenance() const noexcept {
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

    constexpr CertifiedRef(const T &value, EvidenceProvenance provenance) noexcept
        : value_(&value), provenance_(provenance) {}

    const T *value_;
    EvidenceProvenance provenance_;
};

namespace detail {

struct evidence_access final {
    template <class T, class... Properties>
    [[nodiscard]] static constexpr CertifiedRef<T, Properties...>
    make(const T &value, EvidenceProvenance provenance) noexcept {
        return CertifiedRef<T, Properties...>(value, provenance);
    }
};

} // namespace detail

template <class T, class... Ps>
struct model_of<CertifiedRef<T, Ps...>> : model_of<T> {};

namespace detail {

template <class T, class... Ps>
struct scalar_of<CertifiedRef<T, Ps...>, void> : scalar_of<T> {};

template <class T, class... Ps>
struct domain_of<CertifiedRef<T, Ps...>, void> : domain_of<T> {};

template <class T, class... Ps>
struct codomain_of<CertifiedRef<T, Ps...>, void> : codomain_of<T> {};

} // namespace detail

/// Attach caller-supplied evidence. Decidable shape prerequisites remain enforced.
template <class P, class T>
requires MathematicalProposition<P> [[nodiscard]] CertifiedRef<T, P>
assume(const T &value, std::source_location location = std::source_location::current()) {
    if constexpr (std::derived_from<P, axiom::square_operator>) {
        if (value.rows() != value.cols()) {
            throw std::invalid_argument(
                "cannot certify a self-adjoint property on a non-square value");
        }
    }
    return detail::evidence_access::make<T, P>(
        value, {evidence_origin::assumed, location, "explicit caller assumption"});
}

// CertifiedRef is non-owning.  Binding it to a temporary would leave dangling
// evidence at the end of the full expression, so rvalues are rejected even
// though a const reference could otherwise bind to them.
template <class P, class T>
requires MathematicalProposition<P> CertifiedRef<T, P>
assume(const T &&, std::source_location = std::source_location::current()) = delete;

/// Exhaustively validate P using a type-specific validator before attaching evidence.
template <class P, class T>
requires MathematicalProposition<P> &&evidence_validator<T, P>::available
    [[nodiscard]] CertifiedRef<T, P>
    require(const T &value, std::source_location location = std::source_location::current()) {
    if (!evidence_validator<T, P>::verify(value)) {
        throw std::invalid_argument("value does not satisfy the required mathematical proposition");
    }
    return detail::evidence_access::make<T, P>(
        value, {evidence_origin::verified, location, "exhaustive validator"});
}

template <class P, class T>
requires MathematicalProposition<P> &&evidence_validator<T, P>::available CertifiedRef<T, P>
require(const T &&, std::source_location = std::source_location::current()) = delete;

} // namespace num::math

namespace num {
using math::assume;
using math::require;
} // namespace num
