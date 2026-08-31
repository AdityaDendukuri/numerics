/// @file 14_concepts_and_property_invariants.cpp
/// @brief Demonstrates C++20 linear algebra concepts, compile-time invariant tagging (assume_spd), and loud warnings on untagged storage.
#include <iomanip>
#include <iostream>
#include <numerics.hpp>

int main() {
    using namespace num;

    std::cout << "========================================================================\n";
    std::cout << "  14: C++20 Linear Algebra Invariants & Loud Diagnostics Demonstration  \n";
    std::cout << "========================================================================\n\n";

    // -------------------------------------------------------------------------
    // 1. Construct a 4x4 Symmetric Positive-Definite (SPD) 1D Laplace Matrix
    // -------------------------------------------------------------------------
    Matrix A(4, 4, 0.0);
    A(0, 0) = 2.0; A(0, 1) = -1.0;
    A(1, 0) = -1.0; A(1, 1) = 2.0; A(1, 2) = -1.0;
    A(2, 1) = -1.0; A(2, 2) = 2.0; A(2, 3) = -1.0;
    A(3, 2) = -1.0; A(3, 3) = 2.0;

    Vector b{1.0, 2.0, 2.0, 1.0};

    // -------------------------------------------------------------------------
    // 2. Untagged input does not compile; the escape hatch is explicit
    // -------------------------------------------------------------------------
    std::cout << "--- [Case 1: Deliberate opt-out via num::unsafe::cholesky(A)] ---\n";
    std::cout << "cholesky(assume_spd(A)) on a raw Matrix is a compile error: a raw matrix carries no SPD\n"
                 "invariant, and Cholesky is undefined without one. Uncommenting the line below\n"
                 "fails to build, rather than warning and continuing:\n\n"
                 "    // auto bad = cholesky(assume_spd(A));  // error: static assertion failed\n\n"
                 "To take the precondition on faith anyway, say so at the call site:\n";

    auto chol_untagged = unsafe::cholesky(A); // opt-out, greppable, no verification
    Vector x_untagged(4, 0.0);
    cholesky_solve(chol_untagged, b, x_untagged);
    std::cout << "unsafe:: Solution x = [" << x_untagged[0] << ", " << x_untagged[1] << ", "
              << x_untagged[2] << ", " << x_untagged[3] << "]\n\n";

    // -------------------------------------------------------------------------
    // 3. Demonstration of Tagged Input via assume_spd(A) -> 100% Warning-Free
    // -------------------------------------------------------------------------
    std::cout << "--- [Case 2: Tagged Invariant Input cholesky(assume_spd(A))] ---\n";
    std::cout << "Wrapping with assume_spd(A) satisfies SPDMatrixLike and runs 100% warning-free:\n";

    auto spd_matrix = assume_spd(A);

    // Static concept verification at compile time:
    static_assert(SPDMatrixLike<decltype(spd_matrix)>, "Must satisfy SPDMatrixLike concept");
    static_assert(SymmetricMatrixLike<decltype(spd_matrix)>, "Must satisfy SymmetricMatrixLike concept");
    static_assert(SquareMatrixLike<decltype(spd_matrix)>, "Must satisfy SquareMatrixLike concept");

    auto chol_tagged = cholesky(spd_matrix);
    Vector x_tagged(4, 0.0);
    cholesky_solve(chol_tagged, b, x_tagged);
    std::cout << "Tagged Solution x   = [" << x_tagged[0] << ", " << x_tagged[1] << ", "
              << x_tagged[2] << ", " << x_tagged[3] << "]\n\n";

    // -------------------------------------------------------------------------
    // 4. Demonstration of Dynamic Invariant Validation via make_spd(A)
    // -------------------------------------------------------------------------
    std::cout << "--- [Case 3: Dynamic Validation via make_spd(A)] ---\n";
    try {
        auto validated = make_spd(A);
        std::cout << "make_spd(A) successfully validated positive definiteness.\n";
    } catch (const std::exception &e) {
        std::cout << "Validation error: " << e.what() << "\n";
    }

    Matrix Indefinite(2, 2, 0.0);
    Indefinite(0, 0) = 1.0; Indefinite(0, 1) = 3.0;
    Indefinite(1, 0) = 3.0; Indefinite(1, 1) = 1.0; // det = -8 < 0
    std::cout << "Testing make_spd on Indefinite Matrix [[1, 3], [3, 1]]...\n";
    try {
        auto invalid = make_spd(Indefinite);
        (void)invalid;
    } catch (const std::exception &e) {
        std::cout << "[Caught Expected Violation] " << e.what() << "\n\n";
    }

    // -------------------------------------------------------------------------
    // 5. In-Terminal Solution Visualization
    // -------------------------------------------------------------------------
    std::vector<double> grid{0.0, 1.0, 2.0, 3.0};
    std::vector<double> sol{x_tagged[0], x_tagged[1], x_tagged[2], x_tagged[3]};

    plt::plot(grid, sol, "x_solution", "linespoints");
    plt::title("14 Concepts & Property Invariants: 1D Poisson Solution");
    plt::xlabel("Grid Node i");
    plt::ylabel("Solution x_i");
    plt::show_dumb(120, 25);

    std::cout << "\n[SUCCESS] Example 14 completed.\n";
    return 0;
}
