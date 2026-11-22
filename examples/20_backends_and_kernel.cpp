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
#include <io/json.hpp>
#include <numerics.hpp>
#include <vector>

using namespace num;

int main() {
    // The same operation through each backend. Each namespace compiles in every build and falls
    // back to the portable loop when its library is absent.
    const idx n = 1 << 16;
    vec<real> x(n), y(n);
    for (idx i = 0; i < n; ++i) {
        x[i] = std::sin(static_cast<real>(i));
        y[i] = std::cos(static_cast<real>(i));
    }
    io::json results = {
        {"backends", {{"blas", has_blas}, {"lapack", has_lapack},
                       {"lapack_default", lapack_default}, {"openmp", has_omp},
                       {"simd", has_simd}}},
        {"lapack_threshold", lapack_factor_threshold},
        {"dot", {{"default", dot(x, y)}, {"sequential", seq::dot(x, y)},
                 {"blas", blas::dot(x, y)}, {"openmp", omp::dot(x, y)}}}
    };

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
    results["determinant"] = {{"kernel", det(kernel_lu)}, {"lapack", det(lapack_lu)}};

    const idx m = 3;
    const std::vector<double> S{4.0, 2.0, -1.0, 2.0, 5.0, 1.0, -1.0, 1.0, 6.0};
    const std::vector<double> b{1.0, 2.0, 3.0};
    std::vector<double> Sb(m), L(m * m), z(m);
    kernel::matvec(Sb.data(), S.data(), b.data(), m, m);
    results["kernel_matvec"] = {Sb[0], Sb[1], Sb[2]};
    results["kernel_dot"] = kernel::dot(b.data(), b.data(), m);
    if (num::cholesky(L.data(), S.data(), m)) {
        num::cholesky_solve(z.data(), L.data(), b.data(), m);
    }
    results["cholesky_solution"] = {z[0], z[1], z[2]};

    const idx k = 1000;
    std::vector<double> rhs(k, 1.0), u(k, 0.0), work(3 * k);
    auto laplacian = [k](const double *v, double *Lv) {
        for (idx i = 0; i < k; ++i) {
            Lv[i] = (2.0 * v[i]) - (i > 0 ? v[i - 1] : 0.0) - (i + 1 < k ? v[i + 1] : 0.0);
        }
    };
    const auto result = num::cg(laplacian, u.data(), rhs.data(), k, work.data(), 1e-10, 2000);
    results["cg"] = {{"converged", result.converged},
                     {"iterations", result.iterations}, {"midpoint", u[k / 2]}};
    io::write_json(results, "result.json");
}
