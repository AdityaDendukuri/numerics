#include <iostream>
#include <numerics.hpp>

int main() {
    std::cout << "=== Standalone Downstream Project Demo ===" << std::endl;

    // 1. Construct a 3x3 symmetric positive definite (SPD) matrix
    num::Matrix A(3, 3, 0.0);
    A(0, 0) = 4.0;
    A(0, 1) = 1.0;
    A(1, 0) = 1.0;
    A(1, 1) = 4.0;
    A(1, 2) = 1.0;
    A(2, 1) = 1.0;
    A(2, 2) = 4.0;

    num::Vector b{1.0, 2.0, 3.0};

    // 2. Create a dense linear operator wrapper (raw, untagged)
    num::operators::DenseOp Aop(A);

    // Check concepts
    static_assert(num::LinearOperator<decltype(Aop)>);
    static_assert(!num::SPDOperator<decltype(Aop)>);

    std::cout << "[1] Raw DenseOp created.\n";
    std::cout << "    - Satisfies LinearOperator?    YES\n";
    std::cout << "    - Satisfies SPDOperator? NO\n\n";

    // UNCOMMENTING THE LINE BELOW FAILS TO COMPILE:
    // num::LinearSolution s_fail = num::solve(num::LinearProblem{Aop, b}, num::CG{});

    // 3. Attach the SPD property tag using assume_spd()
    auto spd_A = num::operators::assume_spd(Aop);
    static_assert(num::SPDOperator<decltype(spd_A)>);

    std::cout << "[2] Wrapped with assume_spd().\n";
    std::cout << "    - Satisfies SPDOperator? YES!\n\n";

    // 4. Solve Ax = b using Conjugate Gradient (CG)
    num::LinearSolution s = num::solve(num::LinearProblem{spd_A, b}, num::CG{});

    std::cout << "[3] Solved Ax = b using Conjugate Gradient (CG):\n";
    std::cout << "    - Converged:     " << (s.converged ? "YES" : "NO") << "\n";
    std::cout << "    - Iterations:    " << s.iterations << "\n";
    std::cout << "    - Residual norm: " << s.residual << "\n";
    std::cout << "    - Solution x:    [" << s.u[0] << ", " << s.u[1] << ", " << s.u[2] << "]\n";

    return 0;
}
