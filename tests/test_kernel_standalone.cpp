/// @file tests/test_kernel_standalone.cpp
/// @brief Proves the tier-0 kernel is usable on its own.
///
/// Compiled against a copy of include/kernel and nothing else, and linked against
/// no library. If a kernel header acquires a dependency -- on a container, on the
/// algebra, on a backend symbol reachable from a constructor -- this stops
/// building, which is the only reliable way to keep the tier copyable.
///
/// Everything below uses the consumer's own storage, never num::vec<real>.

#include "kernel/kernel.hpp"

#include <cmath>
#include <cstdio>
#include <vector>

namespace {

int failures = 0;

void check(bool ok, const char *what) {
    if (!ok) {
        std::printf("FAIL: %s\n", what);
        ++failures;
    }
}

void level1() {
    std::vector<float> x{1, 2, 3, 4}, y{1, 1, 1, 1};
    num::kernel::axpy(y.data(), x.data(), 2.0F, 4);
    check(std::abs(y[3] - 9.0F) < 1e-6F, "axpy over float");
    check(std::abs(num::kernel::dot(x.data(), y.data(), 4) - 70.0F) < 1e-4F, "dot over float");

    const std::vector<float> z{4, 3, 2, 1};
    const auto dots = num::kernel::dot2(x.data(), y.data(), z.data(), 4);
    check(std::abs(dots.xy - 70.0F) < 1e-4F && std::abs(dots.xz - 20.0F) < 1e-4F,
          "two reductions share one traversal");
    const float updated_norm = num::kernel::axpy_norm_sq(y.data(), x.data(), -2.0F, 4);
    check(std::abs(updated_norm - 4.0F) < 1e-4F, "fused update and norm");
}

void block_kernels() {
    using num::idx;
    std::vector<double> A{1, 2, 3, 4, 5, 6};
    std::vector<double> B{7, 8, 9, 10, 11, 12};
    std::vector<double> C(4);
    num::kernel::gemm(C.data(), A.data(), B.data(), 1.0, 0.0, 2, 2, 3);
    check(std::abs(C[0] - 58.0) < 1e-12 && std::abs(C[3] - 154.0) < 1e-12, "dense block product");

    // L * X = RHS, with three right-hand sides stored contiguously by row.
    std::vector<double> L{2, 0, 0, 1, 3, 0, -1, 2, 4};
    std::vector<double> X{2, 4, 6, 7, 11, 15, 9, 18, 27};
    num::kernel::trsm_lower_inplace(X.data(), 3, L.data(), 3, 3);
    std::vector<double> reconstructed(9);
    num::kernel::gemm(reconstructed.data(), L.data(), X.data(), 1.0, 0.0, 3, 3, 3);
    const std::vector<double> rhs{2, 4, 6, 7, 11, 15, 9, 18, 27};
    double worst = 0.0;
    for (idx i = 0; i < rhs.size(); ++i) {
        worst = std::max(worst, std::abs(reconstructed[i] - rhs[i]));
    }
    check(worst < 1e-12, "triangular solve across multiple right-hand sides");

    std::vector<double> U{2, 1, -1, 0, 3, 2, 0, 0, 4};
    std::vector<double> upper_rhs{1, 8, 12};
    num::kernel::trsv_upper(num::kernel::contract::alias_safe, upper_rhs.data(), U.data(),
                                 upper_rhs.data(), 3);
    std::vector<double> upper_check(3);
    num::kernel::matvec(upper_check.data(), U.data(), upper_rhs.data(), 3, 3);
    check(std::abs(upper_check[0] - 1.0) < 1e-12 && std::abs(upper_check[1] - 8.0) < 1e-12 &&
              std::abs(upper_check[2] - 12.0) < 1e-12,
          "upper triangular solve supports an in-place right-hand side");

    // csr matrix [[2,0,1],[0,3,0],[4,0,5]] times a 3x2 dense block.
    std::vector<double> values{2, 1, 3, 4, 5};
    std::vector<idx> row_ptr{0, 2, 3, 5}, col_idx{0, 2, 1, 0, 2};
    std::vector<double> dense{1, 2, 3, 4, 5, 6}, sparse_product(6);
    num::kernel::spmm(sparse_product.data(), 2, values.data(), row_ptr.data(), col_idx.data(),
                           dense.data(), 2, idx(3), 2);
    const std::vector<double> expected{7, 10, 9, 12, 29, 38};
    worst = 0.0;
    for (idx i = 0; i < expected.size(); ++i) {
        worst = std::max(worst, std::abs(sparse_product[i] - expected[i]));
    }
    check(worst < 1e-12, "csr times a dense block");

    std::vector<double> basis{1, 0, 0, 1, 1, 1};
    std::vector<double> vector{2, 3, 4}, coefficients(2);
    num::kernel::project_columns(coefficients.data(), basis.data(), 2, vector.data(), 3, 2);
    check(std::abs(coefficients[0] - 6.0) < 1e-12 && std::abs(coefficients[1] - 7.0) < 1e-12,
          "block projection computes V transpose times a vector");

    std::vector<double> combination(3, 0.0);
    num::kernel::combine_columns(combination.data(), basis.data(), 2, coefficients.data(), 1.0,
                                      0.0, 3, 2);
    check(std::abs(combination[0] - 6.0) < 1e-12 && std::abs(combination[1] - 7.0) < 1e-12 &&
              std::abs(combination[2] - 13.0) < 1e-12,
          "block linear combination computes V times coefficients");


    std::vector<double> transpose_product(4);
    num::kernel::gemm_transpose_left(transpose_product.data(), 2, basis.data(), 2,
                                          basis.data(), 2, 1.0, 0.0, 3, 2, 2);
    const std::vector<double> expected_gram{2, 1, 1, 2};
    worst = 0.0;
    for (idx i = 0; i < expected_gram.size(); ++i) {
        worst = std::max(worst, std::abs(transpose_product[i] - expected_gram[i]));
    }
    check(worst < 1e-12, "transpose-left matrix product computes a Gram matrix");

    const double combination_norm = num::kernel::linear_combination_norm_sq(
        vector.data(), 1.0, combination.data(), -1.0, 3);
    check(std::abs(combination_norm - 113.0) < 1e-12,
          "linear-combination norm avoids materialization");
}

void triangular_products() {
    // L L^T for a hand-written lower factor, rebuilt by syrk and undone by trsv.
    const num::idx n = 3;
    const std::vector<double> L{2, 0, 0, 0.5, 1.5, 0, 0, 1, 2}, b{1, 2, 3};
    std::vector<double> A(n * n, 0.0);
    num::kernel::syrk_lower(A.data(), n, L.data(), n, 1.0, 0.0, n, n);
    const std::vector<double> expected_lower{4, 0, 0, 1, 2.5, 0, 0, 1.5, 5};
    double worst = 0.0;
    for (num::idx i = 0; i < n; ++i) {
        for (num::idx j = 0; j <= i; ++j) {
            worst = std::max(worst, std::abs(A[(i * n) + j] - expected_lower[(i * n) + j]));
        }
    }
    check(worst < 1e-12, "syrk lower forms L L^T");

    std::vector<double> y(n), Ly(n);
    num::kernel::trsv_lower(y.data(), L.data(), b.data(), n);
    num::kernel::matvec(Ly.data(), L.data(), y.data(), n, n);
    worst = 0.0;
    for (num::idx i = 0; i < n; ++i) {
        worst = std::max(worst, std::abs(Ly[i] - b[i]));
    }
    check(worst < 1e-12, "lower triangular solve inverts matvec");
}

} // namespace

int main() {
    level1();
    block_kernels();
    triangular_products();
    if (failures == 0) {
        std::printf("kernel standalone: all checks passed\n");
    }
    return failures == 0 ? 0 : 1;
}
