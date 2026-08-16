/// @file 10_banded_and_spd_operators.cpp
/// @brief Banded Matrix storage (KL, KU), Banded solves, and compile-time SPD operator tags.
#include <numerics.hpp>
#include <iostream>

int main() {
    using namespace num;

    idx n = 5;
    idx kl = 1, ku = 1; // Tridiagonal band structure
    BandedMatrix B(n, kl, ku);

    for (idx i = 0; i < n; ++i) {
        B(i, i) = 4.0;
        if (i > 0) B(i, i - 1) = -1.0;
        if (i + 1 < n) B(i, i + 1) = -1.0;
    }

    Vector x_true(n, 1.0);
    Vector y(n, 0.0);
    banded_matvec(B, x_true, y);

    std::cout << "Banded Matrix (n=" << n << ", kl=" << kl << ", ku=" << ku << ") MatVec y[0] = " << y[0] << "\n";

    // Banded Solve
    Vector x_sol(n, 0.0);
    banded_solve(B, y, x_sol);
    std::cout << "Banded Solve x[0] = " << x_sol[0] << " (True: " << x_true[0] << ")\n";

    // Compile-time SPD property tagging
    Matrix dense_A(n, n, 0.0);
    for (idx i = 0; i < n; ++i) dense_A(i, i) = 4.0;
    operators::DenseOp B_op(dense_A);
    auto spd_tag = operators::assume_spd(B_op);
    static_assert(SPDLinearOperator<decltype(spd_tag), Vector, Vector>);
    std::cout << "Compile-time SPD Property Tag verified.\n";

    return 0;
}
