/// @file test_storage_dispatch.cpp
/// @brief One name per operation, selected by storage layout rather than by concrete type.
///
/// `num::matvec` and friends are constrained on `num::repr::dense_row_major` and
/// `num::repr::csr` rather than declared over `num::mat` and `num::spmat`. The two
/// concepts are disjoint, so the overloads never compete, and any type exposing the
/// accessors participates without an adapter or a trait specialisation.
///
/// The foreign types below are the point of the test: they are the shapes an Eigen sparse
/// matrix or a raw triple of CSR buffers already has. If a future change narrows a
/// constraint back to a concrete numerics type, these stop compiling.

#include "container/matrix_ops.hpp"
#include "linear/matrix_utils.hpp"
#include "linear/sparse/sparse.hpp"
#include <gtest/gtest.h>
#include <vector>

using namespace num;

namespace {

/// Compressed sparse row, owning nothing of numerics.
struct foreign_csr {
    idx nr, nc;
    std::vector<idx> rp, ci;
    std::vector<real> vals;
    [[nodiscard]] idx n_rows() const { return nr; }
    [[nodiscard]] idx n_cols() const { return nc; }
    [[nodiscard]] idx nnz() const { return vals.size(); }
    [[nodiscard]] const idx *row_ptr() const { return rp.data(); }
    [[nodiscard]] const idx *col_idx() const { return ci.data(); }
    [[nodiscard]] const real *values() const { return vals.data(); }
};

/// Row-major dense, likewise.
struct foreign_dense {
    idx r, c;
    std::vector<real> v;
    [[nodiscard]] idx rows() const { return r; }
    [[nodiscard]] idx cols() const { return c; }
    [[nodiscard]] const real *data() const { return v.data(); }
    real *data() { return v.data(); }
    real operator()(idx i, idx j) const { return v[(i * c) + j]; }
};

/// [[2 0 1], [0 3 0], [1 0 4]]
foreign_csr sample_csr() { return {3, 3, {0, 2, 3, 5}, {0, 2, 1, 0, 2}, {2.0, 1.0, 3.0, 1.0, 4.0}}; }

} // namespace

TEST(StorageDispatch, ConceptsClassifyEachLayoutAndDoNotOverlap) {
    static_assert(repr::dense_row_major<mat>);
    static_assert(repr::dense_row_major<foreign_dense>);
    static_assert(repr::csr<spmat>);
    static_assert(repr::csr<foreign_csr>);

    // disjointness is what keeps the overload sets from competing
    static_assert(!repr::csr<mat>);
    static_assert(!repr::csr<foreign_dense>);
    static_assert(!repr::dense_row_major<spmat>);
    static_assert(!repr::dense_row_major<foreign_csr>);
    SUCCEED();
}

TEST(StorageDispatch, OneMatvecServesEveryLayout) {
    const vec ones{1.0, 1.0, 1.0};

    const foreign_csr f = sample_csr();
    vec y(3, 0.0);
    matvec(f, ones, y); // row sums: 3, 3, 5
    EXPECT_NEAR(y[0], 3.0, 1e-12);
    EXPECT_NEAR(y[1], 3.0, 1e-12);
    EXPECT_NEAR(y[2], 5.0, 1e-12);

    foreign_dense d{2, 2, {1.0, 2.0, 3.0, 4.0}};
    const vec u{1.0, 1.0};
    vec w(2, 0.0);
    matvec(d, u, w);
    EXPECT_NEAR(w[0], 3.0, 1e-12);
    EXPECT_NEAR(w[1], 7.0, 1e-12);

    // and the library's own types reach the same name
    const spmat s = spmat::from_triplets(2, 2, {0, 1}, {0, 1}, {5.0, 6.0});
    vec q(2, 0.0);
    matvec(s, u, q);
    EXPECT_NEAR(q[0], 5.0, 1e-12);

    const mat m = identity(2);
    vec t(2, 0.0);
    matvec(m, u, t);
    EXPECT_NEAR(t[0], 1.0, 1e-12);
}

TEST(StorageDispatch, SparseMatvecStillWorksAsAForwarder) {
    const spmat s = spmat::from_triplets(2, 2, {0, 1}, {0, 1}, {5.0, 6.0});
    const vec u{1.0, 1.0};
    vec a(2, 0.0);
    vec b(2, 0.0);
    matvec(s, u, a);
    sparse_matvec(s, u, b);
    EXPECT_EQ(a[0], b[0]);
    EXPECT_EQ(a[1], b[1]);
}

TEST(StorageDispatch, TheOtherOperationsAlsoTakeForeignStorage) {
    const foreign_csr f = sample_csr();
    const foreign_dense d{2, 2, {1.0, 2.0, 3.0, 4.0}};

    const vec dc = diagonal(f);
    EXPECT_NEAR(dc[0], 2.0, 1e-12);
    EXPECT_NEAR(dc[1], 3.0, 1e-12);
    EXPECT_NEAR(dc[2], 4.0, 1e-12);

    const vec dd = diagonal(d);
    EXPECT_NEAR(dd[0], 1.0, 1e-12);
    EXPECT_NEAR(dd[1], 4.0, 1e-12);

    EXPECT_NEAR(dense(f)(0, 2), 1.0, 1e-12);
    EXPECT_NEAR(dense(transpose(f))(2, 0), 1.0, 1e-12);
    EXPECT_NEAR(transpose(d)(0, 1), 3.0, 1e-12);
    EXPECT_NEAR(dense(scaled(f, 2.0))(0, 0), 4.0, 1e-12);
}
