/// @file 20_backends_and_kernel.cpp
/// @brief Backend namespaces, the build's default, and the raw-pointer kernel on plain buffers.
///
/// Every accelerator is a namespace with the same signatures as the kernel, namely `num::seq`
/// (portable), `num::omp`, `num::blas` and `num::lapack`. An untagged call such as `num::dot`
/// resolves at compile time to the build's default. Factorizations choose their own default, and LU
/// uses the kernel up to `lapack_factor_threshold` and an optimized LAPACK above it. `num::kernel`
/// is BLAS on raw pointers, and algorithms such as `num::cholesky` and `num::cg` take raw pointers
/// too.
#include <cmath>
#include <cstdio>
#include <numerics.hpp>
#include <vector>

using namespace num;

int main() {
    std::printf("configured: blas %d, lapack %d, lapack default %d, openmp %d, simd %d\n", has_blas,
                has_lapack, lapack_default, has_omp, has_simd);
    std::printf("LU switches to LAPACK above n = %zu\n\n",
                static_cast<std::size_t>(lapack_factor_threshold));

    // The same operation through each backend. Each namespace compiles in every build and falls
    // back to the portable loop when its library is absent.
    const idx n = 1 << 16;
    vec<real> x(n), y(n);
    for (idx i = 0; i < n; ++i) {
        x[i] = std::sin(static_cast<real>(i));
        y[i] = std::cos(static_cast<real>(i));
    }
    std::printf("dot, default : %.12f\n", dot(x, y));
    std::printf("dot, seq     : %.12f\n", seq::dot(x, y));
    std::printf("dot, blas    : %.12f\n", blas::dot(x, y));
    std::printf("dot, omp     : %.12f\n\n", omp::dot(x, y));

    // A factorization by name: the kernel and LAPACK give the same factors to rounding.
    mat<real> A(200, 200, 0.0);
    for (idx i = 0; i < 200; ++i) {
        for (idx j = 0; j < 200; ++j) {
            A(i, j) = 1.0 / static_cast<real>(1 + i + j);
        }
        A(i, i) += 2.0;
    }
    const lu_result<real> kernel_lu = seq::lu(A);
    const lu_result<real> lapack_lu = lapack::lu(A);
    std::printf("det, kernel LU : %.10e\ndet, LAPACK LU : %.10e\n\n", det(kernel_lu),
                det(lapack_lu));

    const idx m = 3;
    const std::vector<double> S{4.0, 2.0, -1.0, 2.0, 5.0, 1.0, -1.0, 1.0, 6.0};
    const std::vector<double> b{1.0, 2.0, 3.0};
    std::vector<double> Sb(m), L(m * m), z(m);
    kernel::matvec(Sb.data(), S.data(), b.data(), m, m);
    std::printf("kernel matvec S b: %.1f %.1f %.1f, dot(b, b) = %.1f\n", Sb[0], Sb[1], Sb[2],
                kernel::dot(b.data(), b.data(), m));
    if (num::cholesky(L.data(), S.data(), m)) {
        num::cholesky_solve(z.data(), L.data(), b.data(), m);
    }
    std::printf("raw Cholesky solve: %.6f %.6f %.6f\n", z[0], z[1], z[2]);

    const idx k = 1000;
    std::vector<double> rhs(k, 1.0), u(k, 0.0), work(3 * k);
    auto laplacian = [k](const double *v, double *Lv) {
        for (idx i = 0; i < k; ++i) {
            Lv[i] = (2.0 * v[i]) - (i > 0 ? v[i - 1] : 0.0) - (i + 1 < k ? v[i + 1] : 0.0);
        }
    };
    const auto result = num::cg(laplacian, u.data(), rhs.data(), k, work.data(), 1e-10, 2000);
    std::printf("raw CG: converged %d in %zu iterations, u[k/2] = %.4f\n", result.converged,
                static_cast<std::size_t>(result.iterations), u[k / 2]);
}
