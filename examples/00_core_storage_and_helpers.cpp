/// @file 00_core_storage_and_helpers.cpp
/// @brief Vectors, matrices, sparse matrices, expressions and the helpers around them.
///
/// `vec<T>` and `mat<T>` own aligned, contiguous storage, and `mat` is row-major. Arithmetic comes
/// in two forms. Expressions under `num::ops` return new values and suit formulas. Functions with
/// an output parameter, such as `matvec` and `axpy`, allocate nothing and suit loops. `spmat`
/// stores a sparse matrix in CSR form and is built from triplets.
#include <array>
#include <cstdio>
#include <numerics.hpp>
#include <span>
#include <vector>

using namespace num;

namespace {

void print(const char *name, const vec<real> &v) {
    std::printf("%-27s[", name);
    for (idx i = 0; i < v.size(); ++i) {
        std::printf(i ? ", %g" : "%g", v[i]);
    }
    std::printf("]\n");
}

void print(const char *name, const mat<real> &M) {
    for (idx i = 0; i < M.rows(); ++i) {
        std::printf("%-27s[", i ? "" : name);
        for (idx j = 0; j < M.cols(); ++j) {
            std::printf(j ? " %5g" : "%5g", M(i, j));
        }
        std::printf("]\n");
    }
}

} // namespace

int main() {
    // Construction and the in-place forms: nothing below allocates.
    vec<real> x{1.0, 2.0, 3.0};
    vec<real> y(3, 2.0), z(3, 0.0);
    scale(y, 0.5);    // y <- 0.5 y
    add(x, y, z);     // z <- x + y
    axpy(-1.0, x, z); // z <- z - x
    print("x", x);
    print("z = (x + 0.5 y) - x", z);
    std::printf("%-27s%g, %g\n", "dot(x, y), norm(x)", dot(x, y), norm(x));

    mat<real> A(3, 3, 0.0);
    set_diagonal(A, std::array<real, 3>{4.0, 5.0, 6.0});
    A(0, 1) = A(1, 0) = 1.0;
    vec<real> ax(3);
    matvec(A, x, ax);
    print("A", A);
    print("A x", ax);

    // The same arithmetic as expressions, for formulas.
    {
        using namespace num::ops;
        const mat<real> C = A * transpose(A) + 2.0 * identity(3);
        const vec<real> w = A * x - x / 2.0;
        print("A A^T + 2 I", C);
        print("A x - x / 2", w);
    }

    // Constructors for common shapes, and the diagonal.
    print("unit_vector(3, 1)", unit_vector(3, 1));
    print("diagonal(A)", diagonal(A));
    print("linspace(0, 1, 5)", linspace(0.0, 1.0, 5));
    std::printf("%-27s%g\n", "accu(A) (sum of entries)", accu(A));

    // CSR from triplets; duplicates are summed.
    const spmat S = spmat::from_triplets(3, 3, std::vector<idx>{0, 0, 1, 2, 2},
                                         std::vector<idx>{0, 1, 1, 2, 2},
                                         std::vector<real>{2.0, 1.0, 3.0, 4.0, 0.5});
    vec<real> sx(3);
    sparse_matvec(S, x, sx);
    std::printf("%-27s%zu stored entries\n", "S", static_cast<std::size_t>(S.nnz()));
    print("S x", sx);
    print("dense(transpose(S))", dense(transpose(S)));

    // Properties checked exactly, and selection helpers.
    std::printf("%-27ssymmetric %d, SPD %d\n", "A", linear::is_symmetric(A), linear::is_spd(A));
    const vec<real> d = diagonal(A);
    const auto smallest = smallest_indices(std::span<const real>(d.data(), d.size()), 2);
    std::printf("%-27sargmax %zu, two smallest %zu %zu\n", "diagonal(A)",
                static_cast<std::size_t>(argmax(std::span<const real>(d.data(), d.size()))),
                static_cast<std::size_t>(smallest[0]), static_cast<std::size_t>(smallest[1]));
    vec<real> p{0.2, -0.1, 0.8};
    clip_and_normalize_nonnegative(std::span<real>(p.data(), p.size()));
    print("clip and normalize", p);
}
