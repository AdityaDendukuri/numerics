/// @file 00_core_storage_and_helpers.cpp
/// @brief Vectors, matrices and sparse storage in a few operations.
#include <array>
#include <io/sparse_json.hpp>
#include <numerics.hpp>
#include <vector>

using namespace num;

int main() {
    vec<real> x{1.0, 2.0, 3.0};
    vec<real> y(3, 2.0), z(3, 0.0);
    scale(y, 0.5);
    add(x, y, z);
    axpy(-1.0, x, z);

    mat<real> A(3, 3, 0.0);
    set_diagonal(A, std::array<real, 3>{4.0, 5.0, 6.0});
    A(0, 1) = A(1, 0) = 1.0;
    vec<real> Ax(3);
    matvec(A, x, Ax);

    const spmat S = spmat::from_triplets(3, 3,
        std::vector<idx>{0, 0, 1, 2, 2},
        std::vector<idx>{0, 1, 1, 2, 2},
        std::vector<real>{2.0, 1.0, 3.0, 4.0, 0.5});
    vec<real> Sx(3);
    sparse_matvec(S, x, Sx);

    io::write_json({
        {"dot", dot(x, y)}, {"norm", norm(x)},
        {"Ax", {Ax[0], Ax[1], Ax[2]}},
        {"Sx", {Sx[0], Sx[1], Sx[2]}},
        {"sparse_matrix", io::sparse_matrix_json(S)}
    }, "result.json");
}
