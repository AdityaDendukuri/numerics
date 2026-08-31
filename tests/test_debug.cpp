#include "numerics.hpp"
#include <cmath>
#include <gtest/gtest.h>

// Compile-time concept checks
static_assert(num::Scalar<double>);
static_assert(num::Scalar<float>);
static_assert(num::VectorSpace<num::Vector>);
static_assert(num::MutableVectorSpace<num::Vector>);
static_assert(num::repr::Contiguous<num::Vector>);
static_assert(num::MatrixSpace<num::Matrix>);
static_assert(num::MutableMatrixSpace<num::Matrix>);
static_assert(num::repr::DenseRowMajor<num::Matrix>);
static_assert(num::repr::CSR<num::SparseMatrix>);
static_assert(num::Preconditioner<num::JacobiPreconditioner>);
static_assert(num::LinearOperator<num::operators::DenseOp>);
static_assert(num::SelfAdjointOperator<num::operators::SymmetricOp<num::operators::DenseOp>>);
static_assert(num::SPDOperator<num::operators::SPDOp<num::operators::DenseOp>>);

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
    EXPECT_NO_THROW(num::linear::debug::verify_sparse_structure(valid));
}

TEST(DebugCheck, DiagnosticLevels) {
    const auto orig = num::debug::get_level();
    num::debug::set_level(num::debug::DiagnosticLevel::off);
    EXPECT_NO_THROW(num::debug::check_dim(5, 3, "test_vector"));
    num::debug::set_level(orig);
}

TEST(DebugCheck, PresetModesAndScopedGuard) {
    const auto orig = num::get_preset();

    // 1. Unsafe mode disables property error exceptions on non-SPD inputs
    num::set_preset(num::preset::unsafe);
    EXPECT_EQ(num::get_preset(), num::Preset::unsafe);

    num::Matrix A(2, 2, 0.0);
    A(0, 0) = -5.0; // Non-SPD
    A(1, 1) = 1.0;
    num::operators::DenseOp Aop(A);

    // In unsafe mode, assume_spd runs silently without throwing
    EXPECT_NO_THROW(static_cast<void>(num::operators::assume_spd(Aop)));

    // 2. Scoped preset guard
    num::set_preset(num::preset::strict);
    EXPECT_EQ(num::get_preset(), num::Preset::strict);

    {
        num::ScopedPreset guard(num::preset::unsafe);
        EXPECT_EQ(num::get_preset(), num::Preset::unsafe);
        EXPECT_NO_THROW(static_cast<void>(num::operators::assume_spd(Aop)));
    }

    // Restores strict preset automatically on scope exit
    EXPECT_EQ(num::get_preset(), num::Preset::strict);
    EXPECT_THROW(static_cast<void>(num::operators::assume_spd(Aop)), std::invalid_argument);

    num::set_preset(orig);
}
