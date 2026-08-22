/// @file 00_core_storage_and_helpers.cpp
/// @brief Core vector, matrix, sparse, selection, and construction helpers.
#include <array>
#include <iostream>
#include <numerics.hpp>
#include <span>
#include <vector>

int main() {
    using namespace num;

    // Dense vectors own contiguous storage.
    Vector x{1.0, 2.0, 3.0};
    Vector y(3, 2.0);
    Vector z(3, 0.0);

    scale(y, 0.5);    // y <- 0.5 y
    add(x, y, z);     // z <- x+y
    axpy(-1.0, x, z); // z <- z-x
    const real xy = dot(x, y);
    const real length = norm(x);

    // Spans use the same scalar helpers without owning memory.
    const real span_dot =
        dot(std::span<const real>(x.data(), x.size()), std::span<const real>(y.data(), y.size()));
    std::vector<real> host(x.size());
    copy_to(x, host);

    // Interleaved coordinate views avoid copying particle data.
    Vector coordinates{1.0, 2.0, 3.0, 4.0};
    Vec2View points{coordinates};
    points.x(1) = 5.0;

    // Dense matrices are row-major and dispatch arithmetic by backend.
    Matrix A(3, 3, 0.0);
    set_diagonal(A, std::array<real, 3>{4.0, 5.0, 6.0});
    A(0, 1) = 1.0;
    A(1, 0) = 1.0;

    Vector Ax(3, 0.0);
    matvec(A, x, Ax);
    Matrix At = transpose(A);
    Matrix product(3, 3, 0.0);
    matmul(A, At, product);
    Matrix sum(3, 3, 0.0);
    matadd(1.0, A, 1.0, At, sum);

    // Matrix constructors cover common right-hand sides and scalings.
    const Vector e1 = unit_vector(3, 1);
    const Matrix I = identity(3);
    const Matrix rhs = identity_columns(3, 1, 2);
    const Vector diag = diagonal(A);
    const Matrix D = diagonal_matrix(std::span<const real>(diag.data(), diag.size()));

    Vector weighted = x;
    const std::array<real, 3> weights{1.0, 2.0, 4.0};
    scale_elements(weighted, weights);
    divide_elements(weighted, weights);
    scale_rows(A, weights);
    divide_rows(A, weights);

    // Gather and scatter move selected entries by index.
    const std::array<idx, 2> indices{2, 0};
    const auto selected = gather<real>(host, indices);
    std::vector<real> scattered(3, 0.0);
    scatter<real>(selected, indices, scattered);

    // Triplets are sorted and duplicate entries are summed into CSR storage.
    const SparseMatrix sparse = SparseMatrix::from_triplets(3, 3, std::vector<idx>{0, 0, 1, 2},
                                                            std::vector<idx>{0, 1, 1, 2},
                                                            std::vector<real>{2.0, 1.0, 3.0, 4.0});
    Vector sparse_x(3, 0.0);
    sparse_matvec(sparse, x, sparse_x);
    const SparseMatrix sparse_t = transpose(sparse);
    const SparseMatrix half = scaled(sparse, 0.5);
    const Matrix sparse_dense = dense(sparse);
    const Vector sparse_diag = diagonal(sparse);
    const Matrix similar = diagonal_similarity(sparse, weights);

    // Property wrappers distinguish checked and construction-guaranteed claims.
    const bool symmetric = linalg::is_symmetric(A);
    const bool positive_definite = linalg::is_spd(A);
    const auto checked_spd = linalg::make_spd(A);
    const auto assumed_spd = linalg::assume_spd(A);

    // Selection and probability helpers replace common application loops.
    const idx largest = argmax(std::span<const real>(diag.data(), diag.size()));
    const auto smallest = smallest_indices(std::span<const real>(diag.data(), diag.size()), 2);
    Vector probability{0.2, -0.1, 0.8};
    const real clipped_mass =
        clip_and_normalize_nonnegative(std::span<real>(probability.data(), probability.size()));
    const real expectation =
        weighted_sum(std::span<const real>(probability.data(), probability.size()),
                     [&](idx index) { return static_cast<real>(index); });

    std::cout << xy + span_dot + length + points.x(1) + Ax[0] + product(0, 0) + sum(0, 0) + e1[1] +
                     I(0, 0) + rhs(1, 0) + D(0, 0) + scattered[2] + sparse_x[0] + sparse_t(0, 0) +
                     half(0, 0) + sparse_dense(0, 0) + sparse_diag[0] + similar(0, 0) +
                     static_cast<real>(symmetric) + static_cast<real>(positive_definite) +
                     checked_spd.base()(0, 0) + assumed_spd.base()(0, 0) +
                     static_cast<real>(largest + smallest.front()) + clipped_mass + expectation
              << '\n';
}
