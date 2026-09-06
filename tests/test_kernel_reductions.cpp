/// @file tests/test_kernel_reductions.cpp
/// @brief Blocked-accumulation reductions and the overflow-safe Euclidean norm.
///
/// Every reduction in kernel/vector.hpp accumulates in `detail::reduction_lanes`
/// independent chains and finishes with a scalar tail. That shape has two places
/// to get wrong: the boundary between the unrolled body and the tail, and — for
/// the fused kernels that also write — visiting an index twice or not at all.
/// The sizes below straddle the lane count deliberately.

#include "kernel/kernel.hpp"
#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <numeric>
#include <vector>

namespace kern = num::kernel;

namespace {

constexpr num::idx lanes = kern::detail::reduction_lanes;

/// Sizes around the lane boundary, where a mis-written tail shows up.
const std::vector<num::idx> boundary_sizes = {0,         1,         2,         lanes - 1,
                                              lanes,     lanes + 1, 2 * lanes, 2 * lanes + 3,
                                              4 * lanes, 1000};

std::vector<double> ramp(num::idx n, double start = 1.0) {
    std::vector<double> v(n);
    std::iota(v.begin(), v.end(), start);
    return v;
}

TEST(KernelReductions, SumIsExactOnSmallIntegers) {
    for (num::idx n : boundary_sizes) {
        const auto v = ramp(n);
        const double expected = 0.5 * static_cast<double>(n) * (static_cast<double>(n) + 1.0);
        EXPECT_DOUBLE_EQ(kern::sum(v.data(), n), expected) << "n = " << n;
    }
}

TEST(KernelReductions, DotIsExactOnSmallIntegers) {
    for (num::idx n : boundary_sizes) {
        const auto x = ramp(n);
        std::vector<double> y(n, 2.0);
        const double expected = static_cast<double>(n) * (static_cast<double>(n) + 1.0);
        EXPECT_DOUBLE_EQ(kern::dot(x.data(), y.data(), n), expected) << "n = " << n;
    }
}

TEST(KernelReductions, BlockedAndOrderedDotAgree) {
    for (num::idx n : boundary_sizes) {
        const auto x = ramp(n, 0.25);
        const auto y = ramp(n, -3.5);
        const double blocked = kern::dot(x.data(), y.data(), n);
        const double ordered = kern::dot(kern::contract::ordered, x.data(), y.data(), n);
        EXPECT_NEAR(blocked, ordered, 1e-9 * std::max(1.0, std::abs(ordered))) << "n = " << n;
    }
}

TEST(KernelReductions, ThroughputTagStillResolves) {
    // The tag predates the default change; call sites that spell it must keep working.
    const auto x = ramp(37);
    const auto y = ramp(37, 2.0);
    EXPECT_DOUBLE_EQ(kern::dot(kern::contract::throughput, x.data(), y.data(), 37),
                     kern::dot(x.data(), y.data(), 37));
}

TEST(KernelReductions, NormSqAndL1MatchTheDefinition) {
    for (num::idx n : boundary_sizes) {
        const auto v = ramp(n);
        double sq = 0.0;
        double l1 = 0.0;
        for (num::idx i = 0; i < n; ++i) {
            sq += v[i] * v[i];
            l1 += std::abs(v[i]);
        }
        EXPECT_NEAR(kern::norm_sq(v.data(), n), sq, 1e-9 * std::max(1.0, sq)) << "n = " << n;
        EXPECT_NEAR(kern::l1_norm(v.data(), n), l1, 1e-9 * std::max(1.0, l1)) << "n = " << n;
    }
}

TEST(KernelReductions, AxpyNormSqWritesEveryElementExactlyOnce) {
    // The fused kernel reduces through a side-effecting functor. An index visited
    // twice would double-apply the update; one skipped would leave it stale.
    for (num::idx n : boundary_sizes) {
        std::vector<double> y(n, 1.0);
        const std::vector<double> x(n, 2.0);
        const double returned = kern::axpy_norm_sq(y.data(), x.data(), 3.0, n);
        for (num::idx i = 0; i < n; ++i) {
            EXPECT_DOUBLE_EQ(y[i], 7.0) << "n = " << n << ", element " << i;
        }
        EXPECT_NEAR(returned, 49.0 * static_cast<double>(n), 1e-9 * std::max(1.0, 49.0 * n));
    }
}

TEST(KernelReductions, FusedDualReductionsMatchSeparateOnes) {
    for (num::idx n : boundary_sizes) {
        const auto x = ramp(n, 0.5);
        const auto y = ramp(n, 1.5);
        const auto z = ramp(n, -2.5);

        const auto both = kern::dot2(x.data(), y.data(), z.data(), n);
        EXPECT_NEAR(both.xy, kern::dot(x.data(), y.data(), n), 1e-9 * std::max(1.0, both.xy));
        EXPECT_NEAR(both.xz, kern::dot(x.data(), z.data(), n), 1e-9 * std::max(1.0, both.xz));

        const auto dn = kern::dot_norm_sq(x.data(), y.data(), n);
        EXPECT_NEAR(dn.dot, kern::dot(x.data(), y.data(), n), 1e-9 * std::max(1.0, dn.dot));
        EXPECT_NEAR(dn.norm_sq, kern::norm_sq(y.data(), n), 1e-9 * std::max(1.0, dn.norm_sq));
    }
}

// --- the overflow-safe norm -------------------------------------------------

TEST(KernelNorm, MatchesTheDirectFormInTheOrdinaryRange) {
    const std::vector<double> v{3.0, 4.0};
    EXPECT_DOUBLE_EQ(kern::norm(v.data(), 2), 5.0);
}

TEST(KernelNorm, SurvivesElementsWhoseSquaresOverflow) {
    // 1e200 squared is +inf in double; the direct form returns inf for a vector
    // whose true norm is a perfectly representable 2e200.
    const std::vector<double> v(4, 1e200);
    const double result = kern::norm(v.data(), v.size());
    EXPECT_TRUE(std::isfinite(result));
    EXPECT_NEAR(result, 2e200, 1e186);
    EXPECT_TRUE(std::isinf(std::sqrt(kern::norm_sq(v.data(), v.size()))))
        << "guard is testing nothing if the direct form no longer overflows";
}

TEST(KernelNorm, SurvivesElementsWhoseSquaresUnderflow) {
    // 1e-200 squared flushes to zero; the direct form reports a norm of 0 for a
    // vector that is nowhere near zero.
    const std::vector<double> v(4, 1e-200);
    const double result = kern::norm(v.data(), v.size());
    EXPECT_GT(result, 0.0);
    EXPECT_NEAR(result, 2e-200, 1e-214);
    EXPECT_EQ(std::sqrt(kern::norm_sq(v.data(), v.size())), 0.0)
        << "guard is testing nothing if the direct form no longer underflows";
}

TEST(KernelNorm, IsZeroForAZeroVectorAndForNoElements) {
    const std::vector<double> zeros(11, 0.0);
    EXPECT_EQ(kern::norm(zeros.data(), zeros.size()), 0.0);
    EXPECT_EQ(kern::norm(zeros.data(), 0), 0.0);
}

TEST(KernelNorm, PropagatesNonFiniteInput) {
    std::vector<double> with_nan{1.0, std::numeric_limits<double>::quiet_NaN(), 2.0};
    EXPECT_TRUE(std::isnan(kern::norm(with_nan.data(), with_nan.size())));

    std::vector<double> with_inf{1.0, std::numeric_limits<double>::infinity(), 2.0};
    EXPECT_TRUE(std::isinf(kern::norm(with_inf.data(), with_inf.size())));
}

// --- sparse kernels, whose loop structure changed ---------------------------

TEST(KernelSparse, SpmvHandlesEmptyAndRaggedRows) {
    // Rows of length 0, 1, and > lanes in one matrix: the row loop now derives a
    // length and reduces over it, so a zero-length row must produce zero rather
    // than reading past the row pointer.
    const num::idx rows = 4;
    const std::vector<num::idx> row_ptr{0, 0, 1, 4, 4 + lanes + 2};
    std::vector<num::idx> col_idx;
    std::vector<double> values;
    col_idx.push_back(0);
    values.push_back(2.0); // row 1: one entry
    for (int k = 0; k < 3; ++k) {
        col_idx.push_back(static_cast<num::idx>(k));
        values.push_back(1.0); // row 2: three entries
    }
    for (num::idx k = 0; k < lanes + 2; ++k) {
        col_idx.push_back(k % 3);
        values.push_back(0.5); // row 3: straddles the lane boundary
    }
    const std::vector<double> x{1.0, 10.0, 100.0};
    std::vector<double> y(rows, -1.0);

    kern::spmv(y.data(), values.data(), row_ptr.data(), col_idx.data(), x.data(), rows);

    EXPECT_DOUBLE_EQ(y[0], 0.0) << "an empty row must reduce to zero";
    EXPECT_DOUBLE_EQ(y[1], 2.0);
    EXPECT_DOUBLE_EQ(y[2], 111.0);
    double expected_row3 = 0.0;
    for (num::idx k = 0; k < lanes + 2; ++k) {
        expected_row3 += 0.5 * x[k % 3];
    }
    EXPECT_NEAR(y[3], expected_row3, 1e-12);
}

TEST(KernelSparse, SpmvAxpyAppliesScalingAroundTheSameProduct) {
    const num::idx rows = 2;
    const std::vector<num::idx> row_ptr{0, 2, 3};
    const std::vector<num::idx> col_idx{0, 1, 1};
    const std::vector<double> values{1.0, 2.0, 4.0};
    const std::vector<double> x{3.0, 5.0};

    std::vector<double> plain(rows, 0.0);
    kern::spmv(plain.data(), values.data(), row_ptr.data(), col_idx.data(), x.data(), rows);

    std::vector<double> fused{1.0, 1.0};
    kern::spmv_axpy(fused.data(), 2.0, values.data(), row_ptr.data(), col_idx.data(), x.data(), 10.0,
                   rows);

    for (num::idx i = 0; i < rows; ++i) {
        EXPECT_DOUBLE_EQ(fused[i], 2.0 * plain[i] + 10.0 * 1.0) << "row " << i;
    }
}

TEST(KernelDense, MatvecMatchesTheDefinitionAcrossTheLaneBoundary) {
    for (num::idx cols : {num::idx{1}, lanes - 1, lanes, lanes + 1, num::idx{33}}) {
        const num::idx rows = 3;
        std::vector<double> a(rows * cols);
        for (num::idx i = 0; i < rows * cols; ++i) {
            a[i] = static_cast<double>(i % 7) - 3.0;
        }
        const std::vector<double> x(cols, 1.5);
        std::vector<double> y(rows, 0.0);
        kern::matvec(y.data(), a.data(), x.data(), rows, cols);

        for (num::idx i = 0; i < rows; ++i) {
            double expected = 0.0;
            for (num::idx j = 0; j < cols; ++j) {
                expected += a[i * cols + j] * x[j];
            }
            EXPECT_NEAR(y[i], expected, 1e-12) << "cols = " << cols << ", row " << i;
        }
    }
}

} // namespace
