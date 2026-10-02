/// @file test_law_derivation.cpp
/// @brief Laws follow implication, and a projection carries its operand's law onto the
/// subspace without over-claiming.

#include "core/math/laws.hpp"
#include "linear/matrix_utils.hpp"
#include "operator/dense.hpp"
#include "operator/projected.hpp"
#include "operator/properties.hpp"
#include <gtest/gtest.h>

using namespace num;
namespace L = num::law;

namespace {

struct claims_spd_and_dominance {
    using laws = L::list<L::spd, L::diagonally_dominant>;
};

} // namespace

TEST(LawDerivation, AStrongerLawImpliesTheWeakerOnes) {
    static_assert(claims<with_law<mat<real>, L::spd>, L::spd>);
    static_assert(claims<with_law<mat<real>, L::spd>, L::psd>);
    static_assert(claims<with_law<mat<real>, L::spd>, L::self_adjoint>);
    static_assert(!claims<with_law<mat<real>, L::psd>, L::spd>);
    static_assert(!claims<mat<real>, L::self_adjoint>, "a type declaring nothing claims nothing");
    SUCCEED();
}

TEST(LawDerivation, DiagonalDominanceIsIncomparableWithDefiniteness) {
    static_assert(!std::derived_from<L::spd, L::diagonally_dominant>,
                  "[[1, 0.9], [0.9, 1]] is SPD and not diagonally dominant");
    static_assert(!std::derived_from<L::diagonally_dominant, L::self_adjoint>,
                  "a diagonally dominant matrix need not be symmetric");
    static_assert(claims<claims_spd_and_dominance, L::spd>);
    static_assert(claims<claims_spd_and_dominance, L::diagonally_dominant>,
                  "a type may declare incomparable laws together");
    SUCCEED();
}

TEST(LawDerivation, ProjectionCarriesTheLawOntoTheSubspace) {
    const mat<real> identity_4 = identity(4);
    const auto a = num::assume_spd(operators::dense_op(identity_4));
    const auto pa = operators::projected(a, space::zero_sum{});

    // P*A agrees with P*A*P on the subspace, and P*A*P inherits definiteness from A.
    static_assert(claims<decltype(pa), L::spd_on<space::zero_sum>>);
    static_assert(claims<decltype(pa), L::psd_on<space::zero_sum>>);
    static_assert(claims<decltype(pa), L::self_adjoint_on<space::zero_sum>>);
    SUCCEED();
}

TEST(LawDerivation, ProjectionDoesNotClaimTheGlobalLaw) {
    const mat<real> identity_4 = identity(4);
    const auto a = num::assume_spd(operators::dense_op(identity_4));
    const auto pa = operators::projected(a, space::zero_sum{});

    // (P*A)^* = A*P != P*A, so the global law does not hold and must not be claimed.
    static_assert(!claims<decltype(pa), L::self_adjoint>);
    static_assert(!claims<decltype(pa), L::spd>);
    SUCCEED();
}

TEST(LawDerivation, ProjectionOfAWeakerOperandDerivesAWeakerRestriction) {
    const mat<real> identity_4 = identity(4);
    const auto sym = num::assume_symmetric(operators::dense_op(identity_4));
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

TEST(LawDerivation, DiagonalDominanceIsCheckedExactlyNotSampled) {
    mat<real> dominant(3, 3, 0.0);
    for (idx i = 0; i < 3; ++i) {
        dominant(i, i) = 4.0;
        if (i + 1 < 3) {
            dominant(i, i + 1) = 1.0;
            dominant(i + 1, i) = 1.0;
        }
    }
    EXPECT_NO_THROW(static_cast<void>(assume_diagonally_dominant(dominant)));

    // A single offending row is enough, and no probe direction can hide it.
    mat<real> offending = dominant;
    offending(2, 2) = 1.0; // |1| < |1| from the (2,1) entry
    EXPECT_THROW(static_cast<void>(assume_diagonally_dominant(offending)),
                 std::invalid_argument);
}
