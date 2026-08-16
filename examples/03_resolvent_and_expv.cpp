/// @file 03_resolvent_and_expv.cpp
/// @brief Complex resolvent solves (sI - A)^-1 b, OpenMP batch shifts, and Arnoldi expv.
#include <numerics.hpp>
#include <iostream>
#include <vector>

int main() {
    using namespace num;

    Matrix A(2, 2, 0.0);
    A(0,0) = -2.0; A(0,1) = 1.0;
    A(1,0) = 1.0;  A(1,1) = -2.0;
    Vector b{1.0, 0.0};

    // 1. Single-shift Complex Resolvent Solve: x = (s I - A)^{-1} b
    cplx s(1.0, 2.0);
    std::vector<cplx> x_res = resolvent_solve(s, A, b);
    std::cout << "Resolvent Solve (s=1+2i) x[0] = " << x_res[0] << "\n";

    // 2. OpenMP Multi-shift Batched Resolvent Solve
    std::vector<cplx> shifts = {cplx(1.0, 0.0), cplx(2.0, 1.0), cplx(0.5, 3.0)};
    auto batch_x = resolvent_solve_batch(shifts, A, b);
    std::cout << "Batched Resolvent solved " << batch_x.size() << " shift contours.\n";

    // 3. Arnoldi Krylov Subspace Matrix Exponential e^{t A} v
    operators::DenseOp Aop(A);
    Vector v{1.0, 0.0};
    Vector exp_tv = expv(0.5, Aop, v, 20, 1e-8);
    std::cout << "Matrix Exponential e^{0.5 A} v = [" << exp_tv[0] << ", " << exp_tv[1] << "]\n";

    return 0;
}
