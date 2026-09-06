/// @file tests/test_alignment.cpp
/// @brief Storage alignment guarantees of the dense containers.
///
/// The alignment is load-bearing twice over: the SIMD kernels are told about it
/// through `data()`, and the deallocation path passes it back to the aligned
/// `operator delete`. A container that quietly lost the guarantee would still
/// produce correct arithmetic, so nothing else in the suite would catch it.

#include "container/matrix.hpp"
#include "container/util/aligned_storage.hpp"
#include "container/vector.hpp"
#include "container/vector_ops.hpp"
#include <gtest/gtest.h>
#include <limits>
#include <vector>

namespace {

TEST(Alignment, StorageAlignmentIsAUsablePowerOfTwo) {
    EXPECT_GE(num::storage_alignment, alignof(std::max_align_t));
    EXPECT_EQ(num::storage_alignment & (num::storage_alignment - 1), 0u);
}

TEST(Alignment, VectorStorageIsAligned) {
    // Sizes chosen so the byte counts are not themselves multiples of the
    // alignment: an allocator that only happened to round up would pass at
    // n = 64 and fail here.
    for (num::idx n : {1u, 3u, 7u, 15u, 63u, 64u, 65u, 1000u, 4097u}) {
        num::vec v(n, 1.0);
        EXPECT_TRUE(num::is_storage_aligned(v.data()))
            << "unaligned storage for n = " << n;
    }
}

TEST(Alignment, MatrixStorageIsAligned) {
    for (num::idx rows : {1u, 3u, 17u, 64u}) {
        for (num::idx cols : {1u, 5u, 33u, 64u}) {
            num::mat a(rows, cols, 0.5);
            EXPECT_TRUE(num::is_storage_aligned(a.data()))
                << "unaligned storage for " << rows << "x" << cols;
        }
    }
}

TEST(Alignment, SurvivesEveryConstructionPath) {
    num::vec from_size(37);
    num::vec from_fill(37, 2.0);
    num::vec from_list{1.0, 2.0, 3.0};
    const std::vector<double> source(37, 3.0);
    num::vec from_std(source);
    num::vec copied(from_fill);
    num::vec moved(std::move(from_fill));

    EXPECT_TRUE(num::is_storage_aligned(from_size.data()));
    EXPECT_TRUE(num::is_storage_aligned(from_fill.data()));
    EXPECT_TRUE(num::is_storage_aligned(from_list.data()));
    EXPECT_TRUE(num::is_storage_aligned(from_std.data()));
    EXPECT_TRUE(num::is_storage_aligned(copied.data()));
    EXPECT_TRUE(num::is_storage_aligned(moved.data()));

    num::vec assigned(3);
    assigned = copied;
    EXPECT_TRUE(num::is_storage_aligned(assigned.data()));
    EXPECT_EQ(assigned.size(), copied.size());
}

TEST(Alignment, ComplexVectorStorageIsAligned) {
    num::cvec v(19);
    EXPECT_TRUE(num::is_storage_aligned(v.data()));
}

TEST(Alignment, EmptyContainersOwnNoStorage) {
    // data() feeds std::assume_aligned, whose precondition is a pointer to an
    // object; an empty container has to return null rather than a bogus address.
    const num::vec v;
    const num::mat a;
    EXPECT_EQ(v.data(), nullptr);
    EXPECT_EQ(a.data(), nullptr);
    EXPECT_EQ(v.size(), 0u);
    EXPECT_EQ(a.size(), 0u);
}

TEST(Alignment, ValueInitializedStorageIsZeroed) {
    // make_aligned must value-initialize; make_aligned_for_overwrite must not be
    // substituted for it on the sizing constructor.
    const num::vec v(64);
    for (num::idx i = 0; i < v.size(); ++i) {
        EXPECT_EQ(v[i], 0.0) << "element " << i;
    }
    const num::mat a(8, 8);
    for (num::idx i = 0; i < a.rows(); ++i) {
        for (num::idx j = 0; j < a.cols(); ++j) {
            EXPECT_EQ(a(i, j), 0.0) << "element " << i << "," << j;
        }
    }
}

TEST(Alignment, MatrixRejectsAnOverflowingShape) {
    // rows * cols wraps in idx; the wrapped product would allocate a short
    // buffer that operator() then indexes past.
    const num::idx huge = (std::numeric_limits<num::idx>::max() / 2) + 1;
    EXPECT_THROW(num::mat(huge, 4), std::overflow_error);
    EXPECT_THROW(num::mat(4, huge, 1.0), std::overflow_error);
}

TEST(Alignment, AllocationRejectsAnOverflowingByteCount) {
    const num::idx too_many = (std::numeric_limits<num::idx>::max() / sizeof(double)) + 1;
    EXPECT_THROW((void)num::make_aligned<double>(too_many), std::bad_alloc);
    EXPECT_THROW((void)num::make_aligned_for_overwrite<double>(too_many), std::bad_alloc);
}

TEST(Alignment, ArithmeticIsUnchanged) {
    // The alignment is an optimizer hint plus an allocator change; the numbers
    // it produces must be identical.
    num::vec x{1.0, 2.0, 3.0, 4.0};
    const num::vec y{5.0, 6.0, 7.0, 8.0};
    EXPECT_DOUBLE_EQ(num::dot(x, y), 70.0);
    num::axpy(2.0, y, x);
    EXPECT_DOUBLE_EQ(x[0], 11.0);
    EXPECT_DOUBLE_EQ(x[3], 20.0);
}

} // namespace
