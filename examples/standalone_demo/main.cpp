#include <iostream>
#include <numerics.hpp>

int main() {
    std::cout << "=== Standalone Downstream Project Demo ===" << std::endl;

    // 1. Construct a 3x3 symmetric positive definite (SPD) matrix
    num::mat<real> A(3, 3, 0.0);
    A(0, 0) = 4.0;
    A(0, 1) = 1.0;
    A(1, 0) = 1.0;
    A(1, 1) = 4.0;
    A(1, 2) = 1.0;
    A(2, 1) = 1.0;
    A(2, 2) = 4.0;

    num::vec<real> b{1.0, 2.0, 3.0};

    // 2. Create a dense linear operator wrapper (raw, untagged)
    num::operators::dense_op Aop(A);

    // Check concepts
    static_assert(num::linear_operator<decltype(Aop)>);
    static_assert(!num::spd_operator<decltype(Aop)>);

    std::cout << "[1] Raw dense_op created.\n";
    std::cout << "    - Satisfies linear_operator?    YES\n";
    std::cout << "    - Satisfies spd_operator? NO\n\n";

    // UNCOMMENTING THE LINE BELOW FAILS TO COMPILE:
    // num::vec<real> x_fail(3, 0.0);
    // num::cg(Aop, b, x_fail);

    // 3. Attach the SPD property tag using assume_spd()
    auto spd_A = num::assume_spd(Aop);
    static_assert(num::spd_operator<decltype(spd_A)>);

    std::cout << "[2] Wrapped with assume_spd().\n";
    std::cout << "    - Satisfies spd_operator? YES!\n\n";

    // 4. Solve Ax = b using Conjugate Gradient (CG)
    num::vec<real> x(3, 0.0);
    num::solver_result s = num::cg(spd_A, b, x);

    std::cout << "[3] Solved Ax = b using conjugate gradients:\n";
    std::cout << "    - Converged:     " << (s.converged ? "YES" : "NO") << "\n";
    std::cout << "    - Iterations:    " << s.iterations << "\n";
    std::cout << "    - Residual norm: " << s.residual << "\n";
    std::cout << "    - Solution x:    [" << x[0] << ", " << x[1] << ", " << x[2] << "]\n";

    return 0;
}
