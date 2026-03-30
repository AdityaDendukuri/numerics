/// @file linalg/factorization/backends/seq/thomas.cpp
/// @brief Sequential Thomas algorithm (forward elimination + back substitution).
#include "impl.hpp"

namespace num::backends::seq {

void thomas(const Vector& a, const Vector& b, const Vector& c,
            const Vector& d, Vector& x) {
    const idx n = b.size();

    Vector b_work = b;
    Vector d_work = d;

    for (idx i = 1; i < n; ++i) {
        real w = a[i - 1] / b_work[i - 1];
        b_work[i] -= w * c[i - 1];
        d_work[i] -= w * d_work[i - 1];
    }

    x[n - 1] = d_work[n - 1] / b_work[n - 1];
    for (idx i = n - 1; i > 0; --i)
        x[i - 1] = (d_work[i - 1] - c[i - 1] * x[i]) / b_work[i - 1];
}

} // namespace num::backends::seq
