#include "numerics.hpp"
#include <cmath>
#include <gtest/gtest.h>

// Compile-time concept checks
static_assert(num::scalar<double>);
static_assert(num::scalar<float>);
static_assert(num::vector_space<num::vec>);
static_assert(num::mutable_vector_space<num::vec>);
static_assert(num::repr::contiguous<num::vec>);
static_assert(num::matrix_space<num::mat>);
static_assert(num::mutable_matrix_space<num::mat>);
static_assert(num::repr::dense_row_major<num::mat>);
static_assert(num::repr::csr<num::spmat>);
static_assert(num::preconditioner<num::jacobi_preconditioner>);
static_assert(num::linear_operator<num::operators::dense_op>);
static_assert(num::self_adjoint_operator<num::operators::symmetric_op<num::operators::dense_op>>);
static_assert(num::spd_operator<num::operators::spd_op<num::operators::dense_op>>);

TEST(DebugCheck, DimensionMismatch) {
    EXPECT_THROW(num::debug::check_dim(5, 3, "test_vector"), std::invalid_argument);
}

TEST(DebugCheck, NonFiniteValueError) {
    double data[3] = {1.0, NAN, 3.0};
    EXPECT_THROW(num::debug::check_finite(data, 3, "test_array"), std::invalid_argument);
}

TEST(DebugCheck, FalseSPDAssertionCaughtAtRuntime) {
    // Create a 2x2 matrix with negative diagonal entry (indefinite/negative definite)
    num::mat A(2, 2, 0.0);
    A(0, 0) = -5.0;
    A(1, 1) = 1.0;

    num::operators::dense_op Aop(A);

    // assume_spd() throws a PropertyError because sampled x^T A x is <= 0!
    EXPECT_THROW(static_cast<void>(num::operators::assume_spd(Aop)), std::invalid_argument);
}

TEST(DebugCheck, SparseStructureValidation) {
    num::spmat valid = num::spmat::from_triplets(2, 2, {0, 1}, {0, 1}, {1.0, 2.0});
    EXPECT_NO_THROW(num::linear::debug::verify_sparse_structure(valid));
}

TEST(DebugCheck, DiagnosticLevels) {
    const auto orig = num::debug::get_level();
    num::debug::set_level(num::debug::diagnostic_level::off);
    EXPECT_NO_THROW(num::debug::check_dim(5, 3, "test_vector"));
    num::debug::set_level(orig);
}

TEST(DebugCheck, PresetModesAndScopedGuard) {
    const auto orig = num::get_preset();

    // 1. Unsafe mode disables property error exceptions on non-SPD inputs
    num::set_preset(num::preset::unsafe);
    EXPECT_EQ(num::get_preset(), num::diagnostic_preset::unsafe);

    num::mat A(2, 2, 0.0);
    A(0, 0) = -5.0; // Non-SPD
    A(1, 1) = 1.0;
    num::operators::dense_op Aop(A);

    // In unsafe mode, assume_spd runs silently without throwing
    EXPECT_NO_THROW(static_cast<void>(num::operators::assume_spd(Aop)));

    // 2. Scoped preset guard
    num::set_preset(num::preset::strict);
    EXPECT_EQ(num::get_preset(), num::diagnostic_preset::strict);

    {
        num::scoped_preset guard(num::preset::unsafe);
        EXPECT_EQ(num::get_preset(), num::diagnostic_preset::unsafe);
        EXPECT_NO_THROW(static_cast<void>(num::operators::assume_spd(Aop)));
    }

    // Restores strict preset automatically on scope exit
    EXPECT_EQ(num::get_preset(), num::diagnostic_preset::strict);
    EXPECT_THROW(static_cast<void>(num::operators::assume_spd(Aop)), std::invalid_argument);

    num::set_preset(orig);
}
