#include "numerics.hpp"
#include <cmath>
#include <gtest/gtest.h>

// Compile-time concept checks
static_assert(num::Scalar<double>);
static_assert(num::Scalar<float>);
static_assert(num::VectorLike<num::Vector>);
static_assert(num::MutableVectorLike<num::Vector>);
static_assert(num::ContiguousVectorLike<num::Vector>);
static_assert(num::DenseMatrixLike<num::Matrix>);
static_assert(num::MutableDenseMatrixLike<num::Matrix>);
static_assert(num::ContiguousDenseMatrixLike<num::Matrix>);
static_assert(num::SparseMatrixLike<num::SparseMatrix>);
static_assert(num::Preconditioner<num::JacobiPreconditioner>);
static_assert(num::LinearOperator<num::operators::DenseOp>);
static_assert(num::SymmetricLinearOperator<num::operators::SymmetricOp<num::operators::DenseOp>>);
static_assert(num::SPDLinearOperator<num::operators::SPDOp<num::operators::DenseOp>>);

TEST(DebugCheck, DimensionMismatch) {
    EXPECT_THROW(num::debug::check_dim(5, 3, "test_vector"), std::invalid_argument);
}

TEST(DebugCheck, NonFiniteValueError) {
    double data[3] = {1.0, NAN, 3.0};
    EXPECT_THROW(num::debug::check_finite(data, 3, "test_array"), std::invalid_argument);
}

TEST(DebugCheck, FalseSPDAssertionCaughtAtRuntime) {
    // Create a 2x2 matrix with negative diagonal entry (indefinite/negative definite)
    num::Matrix A(2, 2, 0.0);
    A(0, 0) = -5.0;
    A(1, 1) = 1.0;

    num::operators::DenseOp Aop(A);

    // assume_spd() throws a PropertyError because sampled x^T A x is <= 0!
    EXPECT_THROW(static_cast<void>(num::operators::assume_spd(Aop)), std::invalid_argument);
}

TEST(DebugCheck, SparseStructureValidation) {
    num::SparseMatrix valid = num::SparseMatrix::from_triplets(2, 2, {0, 1}, {0, 1}, {1.0, 2.0});
    EXPECT_NO_THROW(num::debug::verify_sparse_structure(valid));
}

TEST(DebugCheck, DiagnosticLevels) {
    const auto orig = num::debug::get_level();
    num::debug::set_level(num::debug::DiagnosticLevel::off);
    EXPECT_NO_THROW(num::debug::check_dim(5, 3, "test_vector"));
    num::debug::set_level(orig);
}
