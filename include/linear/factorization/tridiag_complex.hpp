/// @file linear/factorization/tridiag_complex.hpp
/// @brief Precomputed Thomas solver for constant-coefficient complex systems.
///
/// Solves \f$a x_{k-1}+b x_k+c x_{k+1}=d_k\f$ with fixed \f$a,b,c\f$.
#pragma once

#include "core/types.hpp"
#include <complex>
#include <vector>

namespace num {

/// Reusable factorization of a constant-coefficient complex tridiagonal matrix.
struct ComplexTriDiag {
    using cplx = std::complex<double>;

    std::vector<cplx> c_mod;
    std::vector<cplx> inv_b;
    int n = 0;
    cplx a_coeff = {};

    /// Factor an n-by-n tridiagonal matrix with constant lower/diagonal/upper entries.
    void factor(int n_, cplx a_, cplx b_, cplx c_);

    /// Replace a matching right-hand side with its solution.
    void solve(std::vector<cplx> &d) const;
};



inline void ComplexTriDiag::factor(int n_, cplx a_, cplx b_, cplx c_) {
    n = n_;
    a_coeff = a_;

    c_mod.resize(n);
    inv_b.resize(n);

    inv_b[0] = cplx(1.0, 0.0) / b_;
    c_mod[0] = c_ * inv_b[0];

    for (int k = 1; k < n; ++k) {
        cplx bk = b_ - a_coeff * c_mod[k - 1];
        inv_b[k] = cplx(1.0, 0.0) / bk;
        if (k < n - 1) {
            c_mod[k] = c_ * inv_b[k];
        }
    }
}

inline void ComplexTriDiag::solve(std::vector<cplx> &d) const {
    // Forward sweep
    d[0] *= inv_b[0];
    for (int k = 1; k < n; ++k) {
        d[k] = (d[k] - a_coeff * d[k - 1]) * inv_b[k];
    }

    // Back substitution
    for (int k = n - 2; k >= 0; --k) {
        d[k] -= c_mod[k] * d[k + 1];
    }
}

} // namespace num
