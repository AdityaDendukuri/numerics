#include <iostream>
#include <numerics.hpp>

int main() {
    num::mat<num::real> A(3, 3, 0.0);
    A(0, 0) = A(1, 1) = A(2, 2) = 4.0;
    A(0, 1) = A(1, 0) = A(1, 2) = A(2, 1) = 1.0;

    const num::vec<num::real> b{1.0, 2.0, 3.0};
    const num::operators::dense_op op(A);
    static_assert(num::linear_operator<decltype(op)>);
    static_assert(!num::spd_operator<decltype(op)>);

    num::vec<num::real> x(3, 0.0);
    const auto result = num::cg(num::assume_spd(op), b, x);
    std::cout << "converged: " << result.converged << ", iterations: " << result.iterations
              << ", residual: " << result.residual << ", x: " << x[0] << " " << x[1] << " "
              << x[2] << "\n";
    return result.converged ? 0 : 1;
}
