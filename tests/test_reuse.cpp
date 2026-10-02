/// @file tests/test_reuse.cpp
/// @brief Factoring R again from the factor of a matrix that differs at a few slots.
///
/// Each matrix is a path of slots, and each slot holds a label. Replacing a
/// label changes that slot's row and column, and its diagonal depends only on
/// its own label, as the total rate of a state does. Every solve is compared
/// with a fresh dense LU of the current matrix.

#include "linear/factorization/lu.hpp"
#include "linear/factorization/reuse.hpp"
#include "linear/matrix_utils.hpp"
#include <gtest/gtest.h>

using namespace num;

namespace {

constexpr real tolerance = 1e-10;

real rate(idx from, idx to) {
    return 0.5 + (0.1 * static_cast<real>(((from * 7) + (to * 3)) % 5) / 5.0);
}

mat<real> path(const array<idx> &label) {
    const idx n = label.size();
    mat<real> R(n, n, 0.0);
    for (idx i = 0; i < n; ++i) {
        R(i, i) = 3.0 + (0.1 * static_cast<real>(label[i] % 5));
        if (i > 0) {
            R(i, i - 1) = -rate(label[i], label[i - 1]);
        }
        if (i + 1 < n) {
            R(i, i + 1) = -rate(label[i], label[i + 1]);
        }
    }
    return R;
}

spmat sparse_of(const mat<real> &A) {
    array<idx> rows, columns;
    array<real> values;
    for (idx i = 0; i < A.rows(); ++i) {
        for (idx j = 0; j < A.cols(); ++j) {
            if (A(i, j) != 0.0) {
                rows.push_back(i);
                columns.push_back(j);
                values.push_back(A(i, j));
            }
        }
    }
    return spmat::from_triplets(A.rows(), A.cols(), rows, columns, values);
}

template <class Factor>
void expect_solves(const Factor &Z, const mat<real> &R) {
    const idx n = R.rows();
    vec<real> b(n, 0.0);
    for (idx i = 0; i < n; ++i) {
        b[i] = 1.0 + static_cast<real>(i % 4);
    }
    mat<real> transposed = transpose(R);
    vec<real> expected, expected_transpose;
    lu_solve(lu(R), b, expected);
    lu_solve(lu(transposed), b, expected_transpose);

    const vec<real> x = num::solve(Z, b);
    const vec<real> y = num::solve(transpose(Z), b);
    for (idx i = 0; i < n; ++i) {
        EXPECT_NEAR(x[i], expected[i], tolerance);
        EXPECT_NEAR(y[i], expected_transpose[i], tolerance);
    }
}

array<idx> labels(idx n) {
    array<idx> label(n);
    for (idx i = 0; i < n; ++i) {
        label[i] = i;
    }
    return label;
}

array<idx> pairs(idx n) {
    array<idx> level(n);
    for (idx i = 0; i < n; ++i) {
        level[i] = i / 2;
    }
    return level;
}

} // namespace

TEST(Reuse, DenseCorrectsUntilTheCutoffThenRefactors) {
    array<idx> label = labels(12);
    corrected_lu Z = lu(sparse_of(path(label)), no_pivot, nullptr, {});
    EXPECT_FALSE(Z.reused());

    // Slots 3 and 8 differ from the first factorization, then 3 changes again.
    for (idx slot : {3, 8, 3}) {
        label[slot] += 100;
        Z = lu(sparse_of(path(label)), no_pivot, &Z, array<idx>{slot});
        EXPECT_TRUE(Z.reused());
        expect_solves(Z, path(label));
    }

    // Two more slots make four that differ, beyond the cutoff of three.
    label[0] += 100;
    label[11] += 100;
    Z = lu(sparse_of(path(label)), no_pivot, &Z, array<idx>{0, 11});
    EXPECT_FALSE(Z.reused());
    expect_solves(Z, path(label));

    label[5] += 100;
    Z = lu(sparse_of(path(label)), no_pivot, &Z, array<idx>{5});
    EXPECT_TRUE(Z.reused());
    expect_solves(Z, path(label));
}

TEST(Reuse, DenseWithNothingChangedKeepsTheFactor) {
    const array<idx> label = labels(6);
    const corrected_lu first = lu(sparse_of(path(label)), no_pivot, nullptr, {});
    const corrected_lu second = lu(sparse_of(path(label)), no_pivot, &first, {});
    EXPECT_TRUE(second.reused());
    expect_solves(second, path(label));
}

TEST(Reuse, TheBlockFactorKeepsThePrefixBeforeTheFirstChange) {
    constexpr idx n = 12;
    array<idx> label = labels(n);
    const array<idx> level = pairs(n);
    suffix_block_lu Z = lu(sparse_of(path(label)), blocks(level), nullptr, {});
    EXPECT_FALSE(Z.reused());

    label[9] += 100;
    Z = lu(sparse_of(path(label)), blocks(level), &Z, array<idx>{9});
    EXPECT_EQ(Z.reused_blocks, 4);
    expect_solves(Z, path(label));

    label[0] += 100;
    Z = lu(sparse_of(path(label)), blocks(level), &Z, array<idx>{0});
    EXPECT_FALSE(Z.reused());
    expect_solves(Z, path(label));
}

TEST(Reuse, AChangeOfSizeFactorsFromScratch) {
    const corrected_lu dense = lu(sparse_of(path(labels(8))), no_pivot, nullptr, {});
    const corrected_lu grown = lu(sparse_of(path(labels(10))), no_pivot, &dense, {});
    EXPECT_FALSE(grown.reused());
    expect_solves(grown, path(labels(10)));

    const suffix_block_lu block = lu(sparse_of(path(labels(8))), blocks(pairs(8)), nullptr, {});
    const suffix_block_lu larger = lu(sparse_of(path(labels(10))), blocks(pairs(10)), &block, {});
    EXPECT_FALSE(larger.reused());
    expect_solves(larger, path(labels(10)));
}
