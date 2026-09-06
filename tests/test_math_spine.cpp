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
    using math_laws = num::math::type_list<num::law::spd>;

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
struct claims_of<std::vector<double>> {
    using type = type_list<law::inner_product_space>;
};

template <>
struct claims_of<spine_test::ForeignDiagonal> {
    using type = type_list<law::linear_map>;
};

} // namespace num::math

namespace {

template <class Op>
concept StrictCgCallable = requires(const Op &op, const num::vec &b, num::vec &x) {
    num::cg(op, b, x);
};

template <class Op>
concept StrictMinresCallable = requires(const Op &op, const num::vec &b, num::vec &x) {
    num::minres(op, b, x);
};

template <class Op, class M>
concept StrictPcgCallable =
    requires(const Op &op, const M &preconditioner, const num::vec &b, num::vec &x) {
    num::pcg(op, preconditioner, b, x);
};

template <class Op, class M>
concept ZeroSumPcgCallable = requires(const Op &op, const M &preconditioner, const num::vec &b,
                                      num::vec &x, const num::space::zero_sum &subspace) {
    num::pcg(op, preconditioner, b, x, subspace);
};

template <class T>
concept CanAssumeTemporarySpd = requires {
    num::assume<num::law::spd>(T{});
};

template <class T>
concept CanRequireTemporarySpd = requires {
    num::require<num::law::spd>(T{});
};

static_assert(num::math::field<double>);
static_assert(num::math::inner_product_space<num::vec>);
static_assert(num::math::inner_product_space<std::vector<double>>);
static_assert(num::math::linear_operator<spine_test::ForeignDiagonal>);
static_assert(num::claims<spine_test::ForeignDiagonal, num::law::spd>);
static_assert(num::math::linear_operator<num::operators::backward_euler_2d>);
static_assert(num::claims<num::operators::backward_euler_2d, num::law::spd>);
static_assert(!StrictCgCallable<num::operators::dense_op>);
static_assert(!StrictCgCallable<num::mat>);
static_assert(!StrictMinresCallable<num::operators::dense_op>);
static_assert(!StrictPcgCallable<num::operators::dense_op, num::jacobi_preconditioner>);
static_assert(StrictPcgCallable<num::operators::backward_euler_2d, num::jacobi_preconditioner>);
static_assert(
    !std::constructible_from<num::math::certified_ref<num::mat, num::law::spd>,
                             const num::mat &>);
static_assert(!CanAssumeTemporarySpd<num::mat>);
static_assert(!CanRequireTemporarySpd<num::mat>);
static_assert(num::math::cpo_detail::tag_invocable<num::math::scale_t, double, num::vec &>);
static_assert(num::math::cpo_detail::tag_invocable<num::math::axpy_t, double, const num::vec &,
                                                  num::vec &>);
static_assert(num::math::cpo_detail::tag_invocable<num::math::inner_t, const num::vec &,
                                                  const num::vec &>);
static_assert(num::math::cpo_detail::tag_invocable<num::math::norm_t, const num::vec &>);

TEST(MathSpine, VerifiedEvidenceIsNonOwningAndImmutable) {
    num::mat A(2, 2, 0.0);
    A(0, 0) = 2.0;
    A(1, 1) = 3.0;

    const auto proof = num::require<num::law::spd>(A);
    static_assert(num::claims<decltype(proof), num::law::spd>);
    static_assert(std::same_as<decltype(proof.get()), const num::mat &>);
    EXPECT_EQ(&proof.get(), &A);
    EXPECT_EQ(proof.provenance().origin, num::math::evidence_origin::verified);

    const num::math::certified_ref<num::mat, num::law::self_adjoint> weaker = proof;
    EXPECT_EQ(&weaker.get(), &A);
    EXPECT_EQ(weaker.provenance().origin, num::math::evidence_origin::verified);
}

TEST(MathSpine, AssumeEnforcesDecidableShapePrerequisite) {
    num::mat rectangular(2, 3, 0.0);
    EXPECT_THROW((void)num::assume<num::law::spd>(rectangular),
                 std::invalid_argument);
}

TEST(MathSpine, AssumedEvidenceRecordsItsOrigin) {
    num::mat A(2, 2, 0.0);
    const auto proof = num::assume<num::law::spd>(A);

    EXPECT_EQ(proof.provenance().origin, num::math::evidence_origin::assumed);
    EXPECT_EQ(proof.provenance().location.file_name(), std::string_view(__FILE__));
}

TEST(MathSpine, VerifiedDenseMatrixUsesCanonicalCg) {
    num::mat A(2, 2, 0.0);
    A(0, 0) = 2.0;
    A(1, 1) = 4.0;
    const auto proof = num::require<num::law::spd>(A);
    num::vec b{2.0, 8.0};
    num::vec x(2, 0.0);

    const auto result = num::cg(proof, b, x);

    EXPECT_TRUE(result.converged);
    EXPECT_NEAR(x[0], 1.0, 1e-12);
    EXPECT_NEAR(x[1], 2.0, 1e-12);
}

TEST(MathSpine, CanonicalCgReportsContradictedAssumption) {
    num::mat A(2, 2, 0.0);
    A(0, 0) = -1.0;
    A(1, 1) = -1.0;
    const auto claimed = num::assume<num::law::spd>(A);
    num::vec b{1.0, 1.0};
    num::vec x(2, 0.0);

    EXPECT_THROW((void)num::cg(claimed, b, x), std::runtime_error);
}

TEST(MathSpine, NativeKernelAdapterChecksDimensionsBeforeLowering) {
    const num::vec x(2, 1.0);
    num::vec y(3, 0.0);

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
    num::mat negative(2, 2, 0.0);
    negative(0, 0) = -1.0;
    negative(1, 1) = -1.0;
    const auto claimed = num::assume<num::law::spd>(negative);
    // Use native vectors for the native matrix evidence and verify that the
    // claimed law is checked where the recurrence depends on it.
    num::vec native_b{1.0, 1.0};
    num::vec native_x(2, 0.0);
    num::mat native_A(2, 2, 0.0);
    native_A(0, 0) = 2.0;
    native_A(1, 1) = 4.0;
    const auto certified_A = num::require<num::law::spd>(native_A);
    EXPECT_THROW((void)num::pcg(certified_A, claimed, native_b, native_x), std::runtime_error);
}

TEST(MathSpine, RestrictedPcgCarriesSubspaceSpecificEvidence) {
    num::mat laplacian(2, 2, 0.0);
    laplacian(0, 0) = 1.0;
    laplacian(0, 1) = -1.0;
    laplacian(1, 0) = -1.0;
    laplacian(1, 1) = 1.0;
    num::mat identity(2, 2, 0.0);
    identity(0, 0) = 1.0;
    identity(1, 1) = 1.0;

    const auto restricted_A =
        num::assume<num::law::spd_on<num::space::zero_sum>>(laplacian);
    const auto restricted_M =
        num::assume<num::law::spd_on<num::space::zero_sum>>(identity);
    static_assert(num::claims<decltype(restricted_A),
                                     num::law::spd_on<num::space::zero_sum>>);
    static_assert(!num::claims<decltype(restricted_A), num::law::spd>);
    static_assert(ZeroSumPcgCallable<decltype(restricted_A), decltype(restricted_M)>);

    num::vec b{1.0, -1.0};
    num::vec x(2, 0.0);
    const auto result = num::pcg(restricted_A, restricted_M, b, x, num::space::zero_sum{},
                                 {.tolerance = 1e-12, .max_iterations = 10});

    EXPECT_TRUE(result.converged);
    EXPECT_NEAR(x[0], 0.5, 1e-12);
    EXPECT_NEAR(x[1], -0.5, 1e-12);
    EXPECT_TRUE(num::math::contains(num::space::zero_sum{}, x));
}

TEST(MathSpine, RestrictedPcgRejectsInputOutsideSubspace) {
    num::mat identity(2, 2, 0.0);
    identity(0, 0) = 1.0;
    identity(1, 1) = 1.0;
    const auto restricted =
        num::assume<num::law::spd_on<num::space::zero_sum>>(identity);
    num::vec incompatible_rhs{1.0, 0.0};
    num::vec x(2, 0.0);

    EXPECT_THROW(
        (void)num::pcg(restricted, restricted, incompatible_rhs, x, num::space::zero_sum{}),
        std::invalid_argument);
}

TEST(MathSpine, RestrictedPcgChecksSubspacePreservation) {
    num::mat identity(2, 2, 0.0);
    identity(0, 0) = 1.0;
    identity(1, 1) = 1.0;
    num::mat bad_preconditioner(2, 2, 0.0);
    bad_preconditioner(0, 0) = 1.0;
    bad_preconditioner(1, 1) = 2.0;
    const auto restricted_A =
        num::assume<num::law::spd_on<num::space::zero_sum>>(identity);
    const auto contradicted_M =
        num::assume<num::law::spd_on<num::space::zero_sum>>(bad_preconditioner);
    num::vec b{1.0, -1.0};
    num::vec x(2, 0.0);

    EXPECT_THROW((void)num::pcg(restricted_A, contradicted_M, b, x, num::space::zero_sum{}),
                 std::runtime_error);
}

TEST(MathSpine, PdeConstructionCarriesSpdIntoCg) {
    num::operators::backward_euler_2d A(4, 0.1);
    num::vec b(A.rows(), 1.0);
    num::vec x(A.rows(), 0.0);

    const auto result = num::cg(A, b, x, {.tolerance = 1e-11, .max_iterations = 100});

    EXPECT_TRUE(result.converged);
    EXPECT_LT(result.residual, 1e-10);
}

TEST(MathSpine, PdeConstructionRejectsUnsupportedSpdClaim) {
    EXPECT_THROW(num::operators::backward_euler_2d(4, -0.1), std::invalid_argument);
}

} // namespace
