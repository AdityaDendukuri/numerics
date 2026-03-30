/// @file linalg/factorization/backends/seq/lu.cpp
/// @brief Sequential Doolittle LU factorization with partial pivoting.
#include "impl.hpp"
#include <cmath>
#include <algorithm>

namespace num::backends::seq {

LUResult lu(const Matrix& A) {
    constexpr real singular_tol = 1e-14;
    const idx n = A.rows();
    LUResult f;
    f.LU  = A;
    f.piv.resize(n);
    f.singular = false;

    Matrix& M = f.LU;

    for (idx k = 0; k < n; ++k) {

        idx  pivot_row = k;
        real pivot_val = std::abs(M(k, k));
        for (idx i = k + 1; i < n; ++i) {
            real v = std::abs(M(i, k));
            if (v > pivot_val) { pivot_val = v; pivot_row = i; }
        }
        f.piv[k] = pivot_row;

        if (pivot_row != k)
            for (idx j = 0; j < n; ++j)
                std::swap(M(k, j), M(pivot_row, j));

        if (std::abs(M(k, k)) < singular_tol) {
            f.singular = true;
            continue;
        }

        const real inv_ukk = real(1) / M(k, k);
        for (idx i = k + 1; i < n; ++i)
            M(i, k) *= inv_ukk;

        for (idx i = k + 1; i < n; ++i) {
            const real lik = M(i, k);
            for (idx j = k + 1; j < n; ++j)
                M(i, j) -= lik * M(k, j);
        }
    }

    return f;
}

} // namespace num::backends::seq
