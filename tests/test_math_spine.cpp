#include "core/math/math.hpp"
#include "linear/math_adapters.hpp"
#include "linear/solvers/math_cg.hpp"
#include "linear/solvers/math_gmres.hpp"
#include "linear/solvers/math_minres.hpp"
#include "linear/solvers/math_pcg.hpp"
#include "linear/solvers/preconditioner.hpp"
#include "operator/dense.hpp"
#include "pde/grid_operators.hpp"
#include <algorithm>
#include <gtest/gtest.h>
#include <type_traits>
#include <utility>
#include <vector>

namespace spine_test {

struct ForeignDiagonal {
    using domain_type = std::vector<double>;
    using codomain_type = std::vector<double>;
    using math_propositions = num::math::type_list<num::axiom::positive_definite>;

    std::vector<double> diagonal;

    explicit ForeignDiagonal(std::vector<double> values) : diagonal(std::move(values)) {
        if (std::ranges::any_of(diagonal, [](double value) { return !(value > 0.0); })) {
            throw std::invalid_argument("ForeignDiagonal requires a positive diagonal");
        }
    }

    [[nodiscard]] std::size_t rows() const { return diagonal.size(); }
    [[nodiscard]] std::size_t cols() const { return diagonal.size(); }

    void apply(const std::vector<double> &x, std::vector<double> &y) const {
        y.resize(diagonal.size());
        for (std::size_t i = 0; i < diagonal.size(); ++i) {
            y[i] = diagonal[i] * x[i];
        }
    }
};

} // namespace spine_test

namespace num::math {

template <>
struct model_of<std::vector<double>> {
    using laws = type_list<law::inner_product_space>;
};

template <>
struct model_of<spine_test::ForeignDiagonal> {
    using laws = type_list<law::linear_map>;
};

} // namespace num::math

namespace {

template <class Op>
concept StrictCgCallable = requires(const Op &op, const num::Vector &b, num::Vector &x) {
    num::cg(op, b, x);
};

template <class Op>
concept StrictMinresCallable = requires(const Op &op, const num::Vector &b, num::Vector &x) {
    num::minres(op, b, x);
};

template <class Op, class M>
concept StrictPcgCallable =
    requires(const Op &op, const M &preconditioner, const num::Vector &b, num::Vector &x) {
    num::pcg(op, preconditioner, b, x);
};

template <class Op, class M>
concept ZeroSumPcgCallable = requires(const Op &op, const M &preconditioner, const num::Vector &b,
                                      num::Vector &x, const num::space::zero_sum &subspace) {
    num::pcg(op, preconditioner, b, x, subspace);
};

template <class T>
concept CanAssumeTemporarySpd = requires {
    num::assume<num::axiom::positive_definite>(T{});
};

template <class T>
concept CanRequireTemporarySpd = requires {
    num::require<num::axiom::positive_definite>(T{});
};

static_assert(num::math::Field<double>);
static_assert(num::math::InnerProductSpace<num::Vector>);
static_assert(num::math::InnerProductSpace<std::vector<double>>);
static_assert(num::math::LinearOperator<spine_test::ForeignDiagonal>);
static_assert(num::math::Carries<spine_test::ForeignDiagonal, num::axiom::positive_definite>);
static_assert(num::math::LinearOperator<num::operators::BackwardEuler2D>);
static_assert(num::math::Carries<num::operators::BackwardEuler2D, num::axiom::positive_definite>);
static_assert(!StrictCgCallable<num::operators::DenseOp>);
static_assert(!StrictCgCallable<num::Matrix>);
static_assert(!StrictMinresCallable<num::operators::DenseOp>);
static_assert(!StrictPcgCallable<num::operators::DenseOp, num::JacobiPreconditioner>);
static_assert(StrictPcgCallable<num::operators::BackwardEuler2D, num::JacobiPreconditioner>);
static_assert(
    !std::constructible_from<num::math::CertifiedRef<num::Matrix, num::axiom::positive_definite>,
                             const num::Matrix &>);
static_assert(!CanAssumeTemporarySpd<num::Matrix>);
static_assert(!CanRequireTemporarySpd<num::Matrix>);
static_assert(num::math::cpo_detail::TagInvocable<num::math::scale_t, double, num::Vector &>);
static_assert(num::math::cpo_detail::TagInvocable<num::math::axpy_t, double, const num::Vector &,
                                                  num::Vector &>);
static_assert(num::math::cpo_detail::TagInvocable<num::math::inner_t, const num::Vector &,
                                                  const num::Vector &>);
static_assert(num::math::cpo_detail::TagInvocable<num::math::norm_t, const num::Vector &>);

TEST(MathSpine, VerifiedEvidenceIsNonOwningAndImmutable) {
    num::Matrix A(2, 2, 0.0);
    A(0, 0) = 2.0;
    A(1, 1) = 3.0;

    const auto proof = num::require<num::axiom::positive_definite>(A);
    static_assert(num::math::Carries<decltype(proof), num::axiom::positive_definite>);
    static_assert(std::same_as<decltype(proof.get()), const num::Matrix &>);
    EXPECT_EQ(&proof.get(), &A);
    EXPECT_EQ(proof.provenance().origin, num::math::evidence_origin::verified);

    const num::math::CertifiedRef<num::Matrix, num::axiom::self_adjoint> weaker = proof;
    EXPECT_EQ(&weaker.get(), &A);
    EXPECT_EQ(weaker.provenance().origin, num::math::evidence_origin::verified);
}

TEST(MathSpine, AssumeEnforcesDecidableShapePrerequisite) {
    num::Matrix rectangular(2, 3, 0.0);
    EXPECT_THROW((void)num::assume<num::axiom::positive_definite>(rectangular),
                 std::invalid_argument);
}

TEST(MathSpine, AssumedEvidenceRecordsItsOrigin) {
    num::Matrix A(2, 2, 0.0);
    const auto proof = num::assume<num::axiom::positive_definite>(A);

    EXPECT_EQ(proof.provenance().origin, num::math::evidence_origin::assumed);
    EXPECT_EQ(proof.provenance().location.file_name(), std::string_view(__FILE__));
}

TEST(MathSpine, VerifiedDenseMatrixUsesCanonicalCg) {
    num::Matrix A(2, 2, 0.0);
    A(0, 0) = 2.0;
    A(1, 1) = 4.0;
    const auto proof = num::require<num::axiom::positive_definite>(A);
    num::Vector b{2.0, 8.0};
    num::Vector x(2, 0.0);

    const auto result = num::cg(proof, b, x);

    EXPECT_TRUE(result.converged);
    EXPECT_NEAR(x[0], 1.0, 1e-12);
    EXPECT_NEAR(x[1], 2.0, 1e-12);
}

TEST(MathSpine, CanonicalCgReportsContradictedAssumption) {
    num::Matrix A(2, 2, 0.0);
    A(0, 0) = -1.0;
    A(1, 1) = -1.0;
    const auto claimed = num::assume<num::axiom::positive_definite>(A);
    num::Vector b{1.0, 1.0};
    num::Vector x(2, 0.0);

    EXPECT_THROW((void)num::cg(claimed, b, x), std::runtime_error);
}

TEST(MathSpine, NativeKernelAdapterChecksDimensionsBeforeLowering) {
    const num::Vector x(2, 1.0);
    num::Vector y(3, 0.0);

    EXPECT_THROW(num::math::axpy(1.0, x, y), std::invalid_argument);
}

TEST(MathSpine, GenericCgSupportsForeignCertifiedTypes) {
    spine_test::ForeignDiagonal A{{2.0, 4.0, 8.0}};
    std::vector<double> b{2.0, 8.0, 24.0};
    std::vector<double> x(3, 0.0);

    const auto result = num::cg(A, b, x, {.tolerance = 1e-12, .max_iterations = 20});

    EXPECT_TRUE(result.converged);
    EXPECT_NEAR(x[0], 1.0, 1e-10);
    EXPECT_NEAR(x[1], 2.0, 1e-10);
    EXPECT_NEAR(x[2], 3.0, 1e-10);
}

TEST(MathSpine, GenericKrylovFamilySupportsForeignCertifiedTypes) {
    spine_test::ForeignDiagonal A{{2.0, 4.0, 8.0}};
    spine_test::ForeignDiagonal inverse{{0.5, 0.25, 0.125}};
    const std::vector<double> b{2.0, 8.0, 24.0};

    std::vector<double> x_pcg(3, 0.0);
    const auto pcg_result =
        num::pcg(A, inverse, b, x_pcg, {.tolerance = 1e-12, .max_iterations = 20});
    EXPECT_TRUE(pcg_result.converged);

    std::vector<double> x_minres(3, 0.0);
    const auto minres_result =
        num::minres(A, b, x_minres, {.tolerance = 1e-12, .max_iterations = 20});
    EXPECT_TRUE(minres_result.converged);

    std::vector<double> x_gmres(3, 0.0);
    const auto gmres_result =
        num::gmres(A, b, x_gmres, {.tolerance = 1e-12, .max_iterations = 20, .restart = 3});
    EXPECT_TRUE(gmres_result.converged);

    for (const auto &solution : {x_pcg, x_minres, x_gmres}) {
        EXPECT_NEAR(solution[0], 1.0, 1e-10);
        EXPECT_NEAR(solution[1], 2.0, 1e-10);
        EXPECT_NEAR(solution[2], 3.0, 1e-10);
    }
}

TEST(MathSpine, PcgRejectsContradictedPreconditionerEvidence) {
    num::Matrix negative(2, 2, 0.0);
    negative(0, 0) = -1.0;
    negative(1, 1) = -1.0;
    const auto claimed = num::assume<num::axiom::positive_definite>(negative);
    // Use native vectors for the native matrix evidence and verify that the
    // claimed law is checked where the recurrence depends on it.
    num::Vector native_b{1.0, 1.0};
    num::Vector native_x(2, 0.0);
    num::Matrix native_A(2, 2, 0.0);
    native_A(0, 0) = 2.0;
    native_A(1, 1) = 4.0;
    const auto certified_A = num::require<num::axiom::positive_definite>(native_A);
    EXPECT_THROW((void)num::pcg(certified_A, claimed, native_b, native_x), std::runtime_error);
}

TEST(MathSpine, RestrictedPcgCarriesSubspaceSpecificEvidence) {
    num::Matrix laplacian(2, 2, 0.0);
    laplacian(0, 0) = 1.0;
    laplacian(0, 1) = -1.0;
    laplacian(1, 0) = -1.0;
    laplacian(1, 1) = 1.0;
    num::Matrix identity(2, 2, 0.0);
    identity(0, 0) = 1.0;
    identity(1, 1) = 1.0;

    const auto restricted_A =
        num::assume<num::axiom::positive_definite_on<num::space::zero_sum>>(laplacian);
    const auto restricted_M =
        num::assume<num::axiom::positive_definite_on<num::space::zero_sum>>(identity);
    static_assert(num::math::Carries<decltype(restricted_A),
                                     num::axiom::positive_definite_on<num::space::zero_sum>>);
    static_assert(!num::math::Carries<decltype(restricted_A), num::axiom::positive_definite>);
    static_assert(ZeroSumPcgCallable<decltype(restricted_A), decltype(restricted_M)>);

    num::Vector b{1.0, -1.0};
    num::Vector x(2, 0.0);
    const auto result = num::pcg(restricted_A, restricted_M, b, x, num::space::zero_sum{},
                                 {.tolerance = 1e-12, .max_iterations = 10});

    EXPECT_TRUE(result.converged);
    EXPECT_NEAR(x[0], 0.5, 1e-12);
    EXPECT_NEAR(x[1], -0.5, 1e-12);
    EXPECT_TRUE(num::math::contains(num::space::zero_sum{}, x));
}

TEST(MathSpine, RestrictedPcgRejectsInputOutsideSubspace) {
    num::Matrix identity(2, 2, 0.0);
    identity(0, 0) = 1.0;
    identity(1, 1) = 1.0;
    const auto restricted =
        num::assume<num::axiom::positive_definite_on<num::space::zero_sum>>(identity);
    num::Vector incompatible_rhs{1.0, 0.0};
    num::Vector x(2, 0.0);

    EXPECT_THROW(
        (void)num::pcg(restricted, restricted, incompatible_rhs, x, num::space::zero_sum{}),
        std::invalid_argument);
}

TEST(MathSpine, RestrictedPcgChecksSubspacePreservation) {
    num::Matrix identity(2, 2, 0.0);
    identity(0, 0) = 1.0;
    identity(1, 1) = 1.0;
    num::Matrix bad_preconditioner(2, 2, 0.0);
    bad_preconditioner(0, 0) = 1.0;
    bad_preconditioner(1, 1) = 2.0;
    const auto restricted_A =
        num::assume<num::axiom::positive_definite_on<num::space::zero_sum>>(identity);
    const auto contradicted_M =
        num::assume<num::axiom::positive_definite_on<num::space::zero_sum>>(bad_preconditioner);
    num::Vector b{1.0, -1.0};
    num::Vector x(2, 0.0);

    EXPECT_THROW((void)num::pcg(restricted_A, contradicted_M, b, x, num::space::zero_sum{}),
                 std::runtime_error);
}

TEST(MathSpine, PdeConstructionCarriesSpdIntoCg) {
    num::operators::BackwardEuler2D A(4, 0.1);
    num::Vector b(A.rows(), 1.0);
    num::Vector x(A.rows(), 0.0);

    const auto result = num::cg(A, b, x, {.tolerance = 1e-11, .max_iterations = 100});

    EXPECT_TRUE(result.converged);
    EXPECT_LT(result.residual, 1e-10);
}

TEST(MathSpine, PdeConstructionRejectsUnsupportedSpdClaim) {
    EXPECT_THROW(num::operators::BackwardEuler2D(4, -0.1), std::invalid_argument);
}

} // namespace
