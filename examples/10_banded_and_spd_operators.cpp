/// @file 10_banded_and_spd_operators.cpp
/// @brief Banded Matrix storage (KL, KU), Banded solves, and compile-time SPD operator
/// tags.
#include <iostream>
#include <numerics.hpp>

int main() {
    using namespace num;

    idx n = 10;
    idx kl = 1, ku = 1;
    BandedMatrix B(n, kl, ku);

    for (idx i = 0; i < n; ++i) {
        B(i, i) = 4.0;
        if (i > 0) {
            B(i, i - 1) = -1.0;
        }
        if (i + 1 < n) {
            B(i, i + 1) = -1.0;
        }
    }

    Vector b(n, 1.0);
    Vector x_sol(n, 0.0);
    banded_solve(B, b, x_sol);

    std::cout << "Banded Matrix (n=" << n << ", kl=" << kl << ", ku=" << ku
              << ") Solved x[0] = " << x_sol[0] << "\n";

    std::vector<double> grid, sol;
    for (idx i = 0; i < n; ++i) {
        grid.push_back(static_cast<double>(i));
        sol.push_back(x_sol[i]);
    }

    plt::plot(grid, sol, "Banded x", "linespoints");
    plt::title("10 Banded Matrices: Solution Vector x");
    plt::xlabel("Grid Index i");
    plt::ylabel("Solution x_i");
    plt::show_dumb(140, 35);

    return 0;
}
