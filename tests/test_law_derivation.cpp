/// @file test_law_derivation.cpp
/// @brief Laws that survive an operation are derived, not re-asserted.
///
/// A certificate the caller has already established should not have to be re-established
/// after an operation that provably preserves it. Before this, every operation dropped its
/// operand's law and returned something claiming only `law::linear_map`, so a caller had
/// to `assume` again -- paying an O(n^2) probe to recover a fact already in hand, or
/// asserting it unverified.
///
/// The rules encoded here are theorems, so the tests are about the *shape* of the
/// derivation rather than about numerics: that the derived law is exactly the strongest
/// one the theorem supports, and never stronger. Over-claiming is the failure that
/// matters, so most of these are negative.

#include "core/math/models.hpp"
#include "linear/matrix_utils.hpp"
#include "linear/solvers/cg.hpp"
#include "operator/dense.hpp"
#include "operator/projected.hpp"
#include "operator/properties.hpp"
#include "operator/sum.hpp"
#include <gtest/gtest.h>

using namespace num;
namespace L = num::law;

// -----------------------------------------------------------------------------
// The meet: the strongest law two operands share
// -----------------------------------------------------------------------------

TEST(LawDerivation, MeetIsTheGreatestLowerBound) {
    static_assert(std::same_as<L::meet_t<L::spd, L::spd>, L::spd>);
    static_assert(std::same_as<L::meet_t<L::spd, L::psd>, L::psd>);
    static_assert(std::same_as<L::meet_t<L::spd, L::self_adjoint>, L::self_adjoint>);
    // spd and unitary share only normality: nothing implies both.
    static_assert(std::same_as<L::meet_t<L::spd, L::unitary>, L::normal>);
    static_assert(std::same_as<L::meet_t<L::projection, L::spd>, L::psd>);
    static_assert(std::same_as<L::meet_t<L::skew_adjoint, L::unitary>, L::normal>);
    SUCCEED();
}

TEST(LawDerivation, MeetIsCommutative) {
    static_assert(std::same_as<L::meet_t<L::spd, L::unitary>, L::meet_t<L::unitary, L::spd>>);
    static_assert(std::same_as<L::meet_t<L::psd, L::skew_adjoint>, L::meet_t<L::skew_adjoint, L::psd>>);
    static_assert(std::same_as<L::meet_t<L::spd, L::linear_map>, L::meet_t<L::linear_map, L::spd>>);
    SUCCEED();
}

// -----------------------------------------------------------------------------
// Sums
// -----------------------------------------------------------------------------

TEST(LawDerivation, SumOfSPDIsDerivedSPD) {
    const mat identity_4 = identity(4);
    const auto a = operators::assume_spd(operators::dense_op(identity_4));
    const auto b = operators::assume_spd(operators::dense_op(identity_4));
    const auto s = operators::sum(a, b);

    static_assert(spd_operator<decltype(s)>,
                  "the SPD cone is closed under addition; the sum must carry the law");

    // and it is accepted where SPD is required, with no second probe
    vec rhs(4, 1.0);
    vec x(4, 0.0);
    const auto r = cg(s, rhs, x, 1e-12, 100);
    EXPECT_TRUE(r.converged);
    EXPECT_NEAR(x[0], 0.5, 1e-12) << "(I + I) x = 1 has x = 1/2";
}

TEST(LawDerivation, SumNeverClaimsMoreThanBothOperandsHave) {
    const mat identity_4 = identity(4);
    const auto spd = operators::assume_spd(operators::dense_op(identity_4));
    const auto sym = operators::assume_symmetric(operators::dense_op(identity_4));
    const auto uni = operators::assume_orthogonal(operators::dense_op(identity_4));
    const auto bare = operators::dense_op(identity_4);

    const auto with_symmetric = operators::sum(spd, sym);
    static_assert(self_adjoint_operator<decltype(with_symmetric)>);
    static_assert(!spd_operator<decltype(with_symmetric)>,
                  "a self-adjoint operand may be indefinite, so the sum is not definite");

    const auto with_unitary = operators::sum(spd, uni);
    static_assert(normal_operator<decltype(with_unitary)>);
    static_assert(!self_adjoint_operator<decltype(with_unitary)>);

    const auto with_untagged = operators::sum(spd, bare);
    static_assert(!self_adjoint_operator<decltype(with_untagged)>,
                  "an operand claiming nothing collapses the sum to linear_map");
    SUCCEED();
}

// -----------------------------------------------------------------------------
// Projection onto a subspace
// -----------------------------------------------------------------------------

TEST(LawDerivation, ProjectionCarriesTheLawOntoTheSubspace) {
    const mat identity_4 = identity(4);
    const auto a = operators::assume_spd(operators::dense_op(identity_4));
    const auto pa = operators::projected(a, space::zero_sum{});

    // P*A agrees with P*A*P on the subspace, and P*A*P inherits definiteness from A.
    static_assert(claims<decltype(pa), L::spd_on<space::zero_sum>>);
    static_assert(claims<decltype(pa), L::psd_on<space::zero_sum>>);
    static_assert(claims<decltype(pa), L::self_adjoint_on<space::zero_sum>>);
    SUCCEED();
}

TEST(LawDerivation, ProjectionDoesNotClaimTheGlobalLaw) {
    const mat identity_4 = identity(4);
    const auto a = operators::assume_spd(operators::dense_op(identity_4));
    const auto pa = operators::projected(a, space::zero_sum{});

    // (P*A)^* = A*P != P*A, so the global law genuinely does not hold and must not be
    // claimed. This is the assertion that stops the derivation from being wishful.
    static_assert(!claims<decltype(pa), L::self_adjoint>);
    static_assert(!claims<decltype(pa), L::spd>);
    static_assert(!claims<decltype(pa), L::psd>);
    SUCCEED();
}

TEST(LawDerivation, ProjectionOfAWeakerOperandDerivesAWeakerRestriction) {
    const mat identity_4 = identity(4);
    const auto sym = operators::assume_symmetric(operators::dense_op(identity_4));
    const auto ps = operators::projected(sym, space::zero_sum{});
    static_assert(claims<decltype(ps), L::self_adjoint_on<space::zero_sum>>);
    static_assert(!claims<decltype(ps), L::psd_on<space::zero_sum>>,
                  "self-adjointness does not imply semidefiniteness, restricted or not");

    const auto bare = operators::dense_op(identity_4);
    const auto pb = operators::projected(bare, space::zero_sum{});
    static_assert(!claims<decltype(pb), L::self_adjoint_on<space::zero_sum>>,
                  "nothing in means nothing out");
    SUCCEED();
}
