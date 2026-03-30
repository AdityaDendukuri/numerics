/// @file eigen/power.cpp
/// @brief Power iteration, inverse iteration, Rayleigh quotient iteration

#include "linalg/eigen/power.hpp"
#include <cmath>
#include <stdexcept>

namespace num {


PowerResult power_iteration(const Matrix& A,
                             real tol, idx max_iter, Backend backend) {
    constexpr real tiny = 1e-300;
    const idx n = A.rows();
    if (A.cols() != n)
        throw std::invalid_argument("power_iteration: matrix must be square");

    Vector v(n, 0.0);
    v[0] = 1.0;

    real lambda = 0.0;
    PowerResult result{0.0, v, 0, false};

    for (idx iter = 0; iter < max_iter; ++iter) {
        result.iterations = iter + 1;

        Vector w(n);
        matvec(A, v, w, backend);

        real new_lambda = dot(v, w, backend);
        detail::normalise(w);

        real delta = std::abs(new_lambda - lambda);
        lambda = new_lambda;
        v = w;

        if (delta < tol * (std::abs(lambda) + tiny)) {
            result.converged = true;
            break;
        }
    }

    result.eigenvalue  = lambda;
    result.eigenvector = v;
    return result;
}

PowerResult inverse_iteration(const Matrix& A, real sigma,
                               real tol, idx max_iter, Backend backend) {
    constexpr real tiny = 1e-300;
    const idx n = A.rows();
    if (A.cols() != n)
        throw std::invalid_argument("inverse_iteration: matrix must be square");

    // Factorize (A - sigma*I) once
    Matrix M = A;
    for (idx i = 0; i < n; ++i) M(i,i) -= sigma;
    LUResult f = lu(M);

    Vector v(n, 0.0);
    v[0] = 1.0;

    real lambda = 0.0;
    PowerResult result{0.0, v, 0, false};

    for (idx iter = 0; iter < max_iter; ++iter) {
        result.iterations = iter + 1;

        Vector w(n);
        lu_solve(f, v, w);
        detail::normalise(w);

        // Rayleigh quotient as eigenvalue estimate
        Vector Av(n);
        matvec(A, w, Av, backend);
        real new_lambda = dot(w, Av, backend);

        real delta = std::abs(new_lambda - lambda);
        lambda = new_lambda;
        v = w;

        if (delta < tol * (std::abs(lambda) + tiny)) {
            result.converged = true;
            break;
        }
    }

    result.eigenvalue  = lambda;
    result.eigenvector = v;
    return result;
}

PowerResult rayleigh_iteration(const Matrix& A, const Vector& x0,
                                real tol, idx max_iter, Backend backend) {
    const idx n = A.rows();
    if (A.cols() != n)
        throw std::invalid_argument("rayleigh_iteration: matrix must be square");
    if (x0.size() != n)
        throw std::invalid_argument("rayleigh_iteration: x0 size mismatch");

    Vector v = x0;
    detail::normalise(v);

    // Initial Rayleigh quotient
    Vector Av(n);
    matvec(A, v, Av, backend);
    real sigma = dot(v, Av, backend);

    PowerResult result{sigma, v, 0, false};

    for (idx iter = 0; iter < max_iter; ++iter) {
        result.iterations = iter + 1;

        // Factorize (A - sigma*I); fresh each iteration (cubic convergence)
        Matrix M = A;
        for (idx i = 0; i < n; ++i) M(i,i) -= sigma;
        LUResult f = lu(M);

        if (f.singular) break;

        Vector w(n);
        lu_solve(f, v, w);
        detail::normalise(w);

        matvec(A, w, Av, backend);
        real new_sigma = dot(w, Av, backend);

        // Residual: ||A*v - sigma*v||
        real res = 0;
        for (idx i = 0; i < n; ++i) {
            real d = Av[i] - new_sigma * w[i];
            res += d*d;
        }
        res = std::sqrt(res);

        sigma = new_sigma;
        v = w;

        if (res < tol) {
            result.converged = true;
            break;
        }
    }

    result.eigenvalue  = sigma;
    result.eigenvector = v;
    return result;
}

} // namespace num
