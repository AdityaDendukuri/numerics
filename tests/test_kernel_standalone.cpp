/// @file tests/test_kernel_standalone.cpp
/// @brief Proves the tier-0 kernel is usable on its own.
///
/// Compiled against a copy of include/kernel and nothing else, and linked against
/// no library. If a kernel header acquires a dependency -- on a container, on the
/// algebra, on a backend symbol reachable from a constructor -- this stops
/// building, which is the only reliable way to keep the tier copyable.
///
/// Everything below uses the consumer's own storage, never num::Vector.

#include "kernel/factor.hpp"
#include "kernel/krylov.hpp"
#include "kernel/raw.hpp"

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
    num::kernel::raw::axpy(y.data(), x.data(), 2.0F, 4);
    check(std::abs(y[3] - 9.0F) < 1e-6F, "axpy over float");
    check(std::abs(num::kernel::raw::dot(x.data(), y.data(), 4) - 70.0F) < 1e-4F, "dot over float");

    const std::vector<float> z{4, 3, 2, 1};
    const auto dots = num::kernel::raw::dot2(x.data(), y.data(), z.data(), 4);
    check(std::abs(dots.xy - 70.0F) < 1e-4F && std::abs(dots.xz - 20.0F) < 1e-4F,
          "two reductions share one traversal");
    const float updated_norm = num::kernel::raw::axpy_norm_sq(y.data(), x.data(), -2.0F, 4);
    check(std::abs(updated_norm - 4.0F) < 1e-4F, "fused update and norm");
}

void block_kernels() {
    using num::idx;
    std::vector<double> A{1, 2, 3, 4, 5, 6};
    std::vector<double> B{7, 8, 9, 10, 11, 12};
    std::vector<double> C(4);
    num::kernel::raw::gemm(C.data(), A.data(), B.data(), 1.0, 0.0, 2, 2, 3);
    check(std::abs(C[0] - 58.0) < 1e-12 && std::abs(C[3] - 154.0) < 1e-12, "dense block product");

    // L * X = RHS, with three right-hand sides stored contiguously by row.
    std::vector<double> L{2, 0, 0, 1, 3, 0, -1, 2, 4};
    std::vector<double> X{2, 4, 6, 7, 11, 15, 9, 18, 27};
    num::kernel::raw::trsm_lower_inplace(X.data(), 3, L.data(), 3, 3);
    std::vector<double> reconstructed(9);
    num::kernel::raw::gemm(reconstructed.data(), L.data(), X.data(), 1.0, 0.0, 3, 3, 3);
    const std::vector<double> rhs{2, 4, 6, 7, 11, 15, 9, 18, 27};
    double worst = 0.0;
    for (idx i = 0; i < rhs.size(); ++i) {
        worst = std::max(worst, std::abs(reconstructed[i] - rhs[i]));
    }
    check(worst < 1e-12, "triangular solve across multiple right-hand sides");

    std::vector<double> U{2, 1, -1, 0, 3, 2, 0, 0, 4};
    std::vector<double> upper_rhs{1, 8, 12};
    num::kernel::raw::trsv_upper(num::kernel::raw::contract::alias_safe, upper_rhs.data(), U.data(),
                                 upper_rhs.data(), 3);
    std::vector<double> upper_check(3);
    num::kernel::raw::matvec(upper_check.data(), U.data(), upper_rhs.data(), 3, 3);
    check(std::abs(upper_check[0] - 1.0) < 1e-12 && std::abs(upper_check[1] - 8.0) < 1e-12 &&
              std::abs(upper_check[2] - 12.0) < 1e-12,
          "upper triangular solve supports an in-place right-hand side");

    // CSR matrix [[2,0,1],[0,3,0],[4,0,5]] times a 3x2 dense block.
    std::vector<double> values{2, 1, 3, 4, 5};
    std::vector<idx> row_ptr{0, 2, 3, 5}, col_idx{0, 2, 1, 0, 2};
    std::vector<double> dense{1, 2, 3, 4, 5, 6}, sparse_product(6);
    num::kernel::raw::spmm(sparse_product.data(), 2, values.data(), row_ptr.data(), col_idx.data(),
                           dense.data(), 2, idx(3), 2);
    const std::vector<double> expected{7, 10, 9, 12, 29, 38};
    worst = 0.0;
    for (idx i = 0; i < expected.size(); ++i) {
        worst = std::max(worst, std::abs(sparse_product[i] - expected[i]));
    }
    check(worst < 1e-12, "CSR times a dense block");

    std::vector<double> basis{1, 0, 0, 1, 1, 1};
    std::vector<double> vector{2, 3, 4}, coefficients(2);
    num::kernel::raw::project_columns(coefficients.data(), basis.data(), 2, vector.data(), 3, 2);
    check(std::abs(coefficients[0] - 6.0) < 1e-12 && std::abs(coefficients[1] - 7.0) < 1e-12,
          "block projection computes V transpose times a vector");

    std::vector<double> combination(3, 0.0);
    num::kernel::raw::combine_columns(combination.data(), basis.data(), 2, coefficients.data(), 1.0,
                                      0.0, 3, 2);
    check(std::abs(combination[0] - 6.0) < 1e-12 && std::abs(combination[1] - 7.0) < 1e-12 &&
              std::abs(combination[2] - 13.0) < 1e-12,
          "block linear combination computes V times coefficients");

    const std::vector<double> orthogonal_basis{1, 0, 0, 1, 0, 0};
    std::vector<double> orthogonalized{2, 3, 4};
    num::kernel::raw::mgs_columns(orthogonalized.data(), orthogonal_basis.data(), 2, 3, 2);
    check(std::abs(orthogonalized[0]) < 1e-12 && std::abs(orthogonalized[1]) < 1e-12 &&
              std::abs(orthogonalized[2] - 4.0) < 1e-12,
          "matrix-column modified Gram-Schmidt");

    std::vector<double> transpose_product(4);
    num::kernel::raw::gemm_transpose_left(transpose_product.data(), 2, basis.data(), 2,
                                          basis.data(), 2, 1.0, 0.0, 3, 2, 2);
    const std::vector<double> expected_gram{2, 1, 1, 2};
    worst = 0.0;
    for (idx i = 0; i < expected_gram.size(); ++i) {
        worst = std::max(worst, std::abs(transpose_product[i] - expected_gram[i]));
    }
    check(worst < 1e-12, "transpose-left matrix product computes a Gram matrix");

    const double combination_norm = num::kernel::raw::linear_combination_norm_sq(
        vector.data(), 1.0, combination.data(), -1.0, 3);
    check(std::abs(combination_norm - 113.0) < 1e-12,
          "linear-combination norm avoids materialization");
}

void factorizations() {
    const num::idx n = 3;
    std::vector<double> A{4, 1, 0, 1, 3, 1, 0, 1, 5}, L(n * n), b{1, 2, 3}, x(n);
    check(num::kernel::raw::cholesky(L.data(), A.data(), n), "cholesky succeeds on SPD input");
    num::kernel::raw::cholesky_solve(x.data(), L.data(), b.data(), n);

    std::vector<double> residual(n);
    num::kernel::raw::matvec(residual.data(), A.data(), x.data(), n, n);
    double worst = 0.0;
    for (num::idx i = 0; i < n; ++i) {
        worst = std::max(worst, std::abs(residual[i] - b[i]));
    }
    check(worst < 1e-12, "cholesky solve reproduces the right-hand side");

    std::vector<double> syrk_c(3 * 3, 0.0);
    num::kernel::raw::syrk_lower(syrk_c.data(), 3, L.data(), 3, 1.0, 0.0, 3, 3);
    worst = 0.0;
    for (num::idx i = 0; i < 3; ++i) {
        for (num::idx j = 0; j <= i; ++j) {
            worst = std::max(worst, std::abs(syrk_c[(i * 3) + j] - A[(i * 3) + j]));
        }
    }
    check(worst < 1e-12, "syrk lower reconstructs the SPD lower triangle");

    std::vector<double> blocked = A;
    check(num::kernel::raw::cholesky_blocked(blocked.data(), 3, 2),
          "blocked cholesky succeeds on SPD input");
    worst = 0.0;
    for (num::idx i = 0; i < 3; ++i) {
        for (num::idx j = 0; j <= i; ++j) {
            worst = std::max(worst, std::abs(blocked[(i * 3) + j] - L[(i * 3) + j]));
        }
    }
    check(worst < 1e-12, "blocked cholesky agrees with unblocked factorization");

    std::vector<double> LU = A, xlu(n);
    std::vector<num::idx> piv(n);
    check(num::kernel::raw::lu_factor(LU.data(), piv.data(), n), "lu factor is nonsingular");
    num::kernel::raw::lu_solve(xlu.data(), LU.data(), piv.data(), b.data(), n);
    worst = 0.0;
    for (num::idx i = 0; i < n; ++i) {
        worst = std::max(worst, std::abs(xlu[i] - x[i]));
    }
    check(worst < 1e-12, "lu and cholesky agree on an SPD system");

    // Exercise the documented alias-safe contracts directly.
    std::vector<double> inplace_factor = A;
    check(num::kernel::raw::cholesky(inplace_factor.data(), inplace_factor.data(), n),
          "cholesky factorization may overwrite its input");
    std::vector<double> inplace_rhs = b;
    num::kernel::raw::cholesky_solve(inplace_rhs.data(), inplace_factor.data(), inplace_rhs.data(),
                                     n);
    worst = 0.0;
    for (num::idx i = 0; i < n; ++i) {
        worst = std::max(worst, std::abs(inplace_rhs[i] - x[i]));
    }
    check(worst < 1e-12, "cholesky solve supports an in-place right-hand side");

    std::vector<double> inplace_lu_rhs = b;
    num::kernel::raw::lu_solve(inplace_lu_rhs.data(), LU.data(), piv.data(), inplace_lu_rhs.data(),
                               n);
    worst = 0.0;
    for (num::idx i = 0; i < n; ++i) {
        worst = std::max(worst, std::abs(inplace_lu_rhs[i] - x[i]));
    }
    check(worst < 1e-12, "lu solve supports an in-place right-hand side");

    std::vector<double> blocked_lu = A;
    std::vector<num::idx> blocked_piv(n);
    check(num::kernel::raw::lu_factor_blocked(blocked_lu.data(), blocked_piv.data(), n, 2),
          "blocked lu is nonsingular");
    std::vector<double> blocked_x(n);
    num::kernel::raw::lu_solve(blocked_x.data(), blocked_lu.data(), blocked_piv.data(), b.data(), n);
    worst = 0.0;
    for (num::idx i = 0; i < n; ++i) {
        worst = std::max(worst, std::abs(blocked_x[i] - x[i]));
    }
    check(worst < 1e-12, "blocked lu solve agrees with unblocked factorization");

    std::vector<double> batch_A;
    batch_A.insert(batch_A.end(), A.begin(), A.end());
    batch_A.insert(batch_A.end(), A.begin(), A.end());
    bool status[2]{};
    check(num::kernel::raw::cholesky_batched(batch_A.data(), batch_A.data(), n, 2, n * n, status),
          "batched small cholesky succeeds");
    check(status[0] && status[1], "batched cholesky reports per-system status");
}

void krylov() {
    const num::idx n = 64;
    // 1D Laplacian with a shift: SPD, tridiagonal, applied matrix-free.
    auto A = [&](const double *v, double *out) {
        for (num::idx i = 0; i < n; ++i) {
            double s = 2.1 * v[i];
            if (i > 0) {
                s -= v[i - 1];
            }
            if (i + 1 < n) {
                s -= v[i + 1];
            }
            out[i] = s;
        }
    };
    std::vector<double> b(n, 1.0), x(n, 0.0), work(3 * n);
    const auto r = num::kernel::raw::cg(A, x.data(), b.data(), n, work.data(), 1e-12, 500);
    check(r.converged, "cg converges");

    auto M = [&](const double *res, double *z) {
        for (num::idx i = 0; i < n; ++i) {
            z[i] = res[i] / 2.1;
        }
    };
    std::vector<double> xp(n, 0.0), workp(4 * n);
    const auto rp = num::kernel::raw::pcg(A, M, xp.data(), b.data(), n, workp.data(), 1e-12, 500);
    check(rp.converged, "pcg converges");

    double worst = 0.0;
    for (num::idx i = 0; i < n; ++i) {
        worst = std::max(worst, std::abs(x[i] - xp[i]));
    }
    check(worst < 1e-8, "cg and pcg reach the same solution");
}

} // namespace

int main() {
    level1();
    block_kernels();
    factorizations();
    krylov();
    if (failures == 0) {
        std::printf("kernel standalone: all checks passed\n");
    }
    return failures == 0 ? 0 : 1;
}
