/// @file test_math_spine.cpp
/// @brief The concepts accept exactly the types they describe, and the solvers take any type
/// that models them.

#include "core/math/math.hpp"
#include "linear/eigen/jacobi_eig.hpp"
#include "linear/math_adapters.hpp"
#include "linear/matrix_properties.hpp"
#include "linear/solvers/cg.hpp"
#include "linear/solvers/gmres.hpp"
#include "linear/solvers/minres.hpp"
#include "linear/solvers/pcg.hpp"
#include "linear/solvers/preconditioner.hpp"
#include "operator/dense.hpp"
#include "operator/properties.hpp"
#include "pde/grid_operators.hpp"
#include <algorithm>
#include <gtest/gtest.h>
#include <string>
#include <utility>
#include <vector>

namespace spine_test {

/// A diagonal operator on std::vector, written without any numerics type.
struct ForeignDiagonal {
    using domain_type = std::vector<double>;
    using codomain_type = std::vector<double>;
    using laws = num::law::list<num::law::spd>;

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

namespace {

template <class Op>
concept CgCallable = requires(const Op &op, const num::vec &b, num::vec &x) {
    num::cg(op, b, x);
};

template <class Op>
concept MinresCallable = requires(const Op &op, const num::vec &b, num::vec &x) {
    num::minres(op, b, x);
};

template <class Op, class M>
concept PcgCallable =
    requires(const Op &op, const M &preconditioner, const num::vec &b, num::vec &x) {
    num::pcg(op, preconditioner, b, x);
};

template <class Op, class M>
concept ZeroSumPcgCallable = requires(const Op &op, const M &preconditioner, const num::vec &b,
                                      num::vec &x, const num::space::zero_sum &subspace) {
    num::pcg(op, preconditioner, b, x, subspace);
};

using spd_mat = num::with_law<num::mat, num::law::spd>;
using symmetric_mat = num::with_law<num::mat, num::law::self_adjoint>;

// Fields are the floating-point types and their complex counterparts, nothing else.
static_assert(num::field<double>);
static_assert(num::field<std::complex<float>>);
static_assert(!num::field<int>);
static_assert(!num::field<std::complex<int>>);

// Spaces are decided by their operations, so standard containers need no declaration.
static_assert(num::inner_product_space<num::vec>);
static_assert(num::inner_product_space<num::cvec>);
static_assert(num::inner_product_space<std::vector<double>>);
static_assert(!num::vector_space<std::vector<int>>);
static_assert(!num::vector_space<std::string>);
static_assert(!num::vector_space<num::mat>);

// Operators: shape and action are structural, laws are declared.
static_assert(num::linear_operator<num::mat>);
static_assert(num::linear_operator<num::operators::dense_op>);
static_assert(!num::self_adjoint_operator<num::operators::dense_op>);
static_assert(num::spd_operator<spine_test::ForeignDiagonal>);
static_assert(num::spd_operator<num::operators::backward_euler_2d>);
static_assert(num::spd_operator<spd_mat>);
static_assert(num::psd_operator<spd_mat>);
static_assert(num::self_adjoint_operator<symmetric_mat>);
static_assert(!num::psd_operator<symmetric_mat>);

// The solvers take exactly the operators whose law they need.
static_assert(!CgCallable<num::operators::dense_op>);
static_assert(!CgCallable<num::mat>);
static_assert(!CgCallable<symmetric_mat>);
static_assert(CgCallable<spd_mat>);
static_assert(!MinresCallable<num::operators::dense_op>);
static_assert(MinresCallable<symmetric_mat>);
static_assert(MinresCallable<spd_mat>);
static_assert(!PcgCallable<num::operators::dense_op, num::jacobi_preconditioner>);
static_assert(PcgCallable<num::operators::backward_euler_2d, num::jacobi_preconditioner>);

// A value carrying a stronger law converts to one carrying a weaker law, never back.
static_assert(std::convertible_to<spd_mat, symmetric_mat>);
static_assert(!std::convertible_to<symmetric_mat, spd_mat>);
static_assert(!std::convertible_to<num::mat, spd_mat>);

static_assert(num::math::cpo_detail::tag_invocable<num::math::scale_t, double, num::vec &>);
static_assert(
    num::math::cpo_detail::tag_invocable<num::math::inner_t, const num::vec &, const num::vec &>);

num::mat diagonal(std::initializer_list<double> values) {
    num::mat A(values.size(), values.size(), 0.0);
    num::idx i = 0;
    for (double value : values) {
        A(i, i) = value;
        ++i;
    }
    return A;
}

TEST(MathSpine, AssumeRejectsANonSquareMatrix) {
    num::mat rectangular(2, 3, 0.0);
    EXPECT_THROW((void)num::assume<num::law::spd>(rectangular), std::invalid_argument);
}

TEST(MathSpine, AssumeSamplesTheClaim) {
    EXPECT_THROW((void)num::assume_spd(diagonal({1.0, -1.0})), std::invalid_argument);
    EXPECT_NO_THROW((void)num::assume_spd(diagonal({1.0, 2.0})));
}

TEST(MathSpine, VerifiedDenseMatrixUsesCg) {
    const auto A = num::make_spd(diagonal({2.0, 4.0}));
    num::vec b{2.0, 8.0};
    num::vec x(2, 0.0);

    const auto result = num::cg(A, b, x);

    EXPECT_TRUE(result.converged);
    EXPECT_NEAR(x[0], 1.0, 1e-12);
    EXPECT_NEAR(x[1], 2.0, 1e-12);
}

TEST(MathSpine, AnSpdMatrixIsAcceptedWhereSymmetryIsRequired) {
    const auto A = num::make_spd(diagonal({3.0, 1.0}));
    const auto eigen = num::eig_sym(A);
    EXPECT_NEAR(eigen.values[0], 1.0, 1e-12);
    EXPECT_NEAR(eigen.values[1], 3.0, 1e-12);
}

TEST(MathSpine, CgReportsAContradictedClaim) {
    const spd_mat claimed(diagonal({-1.0, -1.0}));
    num::vec b{1.0, 1.0};
    num::vec x(2, 0.0);

    EXPECT_THROW((void)num::cg(claimed, b, x), std::runtime_error);
}

TEST(MathSpine, NativeKernelAdapterChecksDimensionsBeforeLowering) {
    const num::vec x(2, 1.0);
    num::vec y(3, 0.0);

    EXPECT_THROW(num::math::axpy(1.0, x, y), std::invalid_argument);
}

TEST(MathSpine, GenericCgSupportsForeignTypes) {
    spine_test::ForeignDiagonal A{{2.0, 4.0, 8.0}};
    std::vector<double> b{2.0, 8.0, 24.0};
    std::vector<double> x(3, 0.0);

    const auto result = num::cg(A, b, x, {.tolerance = 1e-12, .max_iterations = 20});

    EXPECT_TRUE(result.converged);
    EXPECT_NEAR(x[0], 1.0, 1e-10);
    EXPECT_NEAR(x[1], 2.0, 1e-10);
    EXPECT_NEAR(x[2], 3.0, 1e-10);
}

TEST(MathSpine, GenericKrylovFamilySupportsForeignTypes) {
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

TEST(MathSpine, PcgReportsAContradictedPreconditionerClaim) {
    const spd_mat claimed(diagonal({-1.0, -1.0}));
    const auto A = num::make_spd(diagonal({2.0, 4.0}));
    num::vec b{1.0, 1.0};
    num::vec x(2, 0.0);
    EXPECT_THROW((void)num::pcg(A, claimed, b, x), std::runtime_error);
}

using zero_sum_spd = num::law::spd_on<num::space::zero_sum>;

TEST(MathSpine, RestrictedPcgSolvesOnTheSubspace) {
    num::mat laplacian(2, 2, 0.0);
    laplacian(0, 0) = 1.0;
    laplacian(0, 1) = -1.0;
    laplacian(1, 0) = -1.0;
    laplacian(1, 1) = 1.0;

    const num::with_law<num::mat, zero_sum_spd> restricted_A(laplacian);
    const num::with_law<num::mat, zero_sum_spd> restricted_M(diagonal({1.0, 1.0}));
    static_assert(num::claims<decltype(restricted_A), zero_sum_spd>);
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
    const num::with_law<num::mat, zero_sum_spd> restricted(diagonal({1.0, 1.0}));
    num::vec incompatible_rhs{1.0, 0.0};
    num::vec x(2, 0.0);

    EXPECT_THROW(
        (void)num::pcg(restricted, restricted, incompatible_rhs, x, num::space::zero_sum{}),
        std::invalid_argument);
}

TEST(MathSpine, RestrictedPcgChecksSubspacePreservation) {
    const num::with_law<num::mat, zero_sum_spd> restricted_A(diagonal({1.0, 1.0}));
    const num::with_law<num::mat, zero_sum_spd> contradicted_M(diagonal({1.0, 2.0}));
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
