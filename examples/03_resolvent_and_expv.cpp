/// @file 03_resolvent_and_expv.cpp
/// @brief Complex resolvent solves (sI - A)^-1 b, OpenMP batch shifts, and Arnoldi expv.
#include <iostream>
#include <numerics.hpp>
#include <vector>

int main() {
    using namespace num;

    mat A(2, 2, 0.0);
    A(0, 0) = -2.0;
    A(0, 1) = 1.0;
    A(1, 0) = 1.0;
    A(1, 1) = -2.0;
    vec b{1.0, 0.0};

    // 1. Single-shift Complex Resolvent Solve
    cplx s(1.0, 2.0);
    std::vector<cplx> x_res = resolvent_solve(s, A, b);
    std::cout << "Resolvent Solve (s=1+2i) x[0] = " << x_res[0] << "\n";

    // 2. Time evolution via expv e^{t A} v
    operators::dense_op Aop(A);
    vec v{1.0, 0.0};
    std::vector<double> t_vec, u0_vec, u1_vec;
    for (int step = 0; step <= 20; ++step) {
        double t = step * 0.1;
        vec exp_tv = expv(t, Aop, v, 20, 1e-8);
        t_vec.push_back(t);
        u0_vec.push_back(exp_tv[0]);
        u1_vec.push_back(exp_tv[1]);
    }

    plt::plot(t_vec, u0_vec, "plot_state u0(t)", "lines");
    plt::plot(t_vec, u1_vec, "plot_state u1(t)", "lines");
    plt::title("03 Resolvent & Expv: mat Exponential Time Trajectory e^{t A} v");
    plt::xlabel("Time t");
    plt::ylabel("Probability plot_state u(t)");
    plt::legend();
    plt::show_dumb(140, 35);

    return 0;
}
