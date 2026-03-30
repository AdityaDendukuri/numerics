/// @file linalg/factorization/backends/seq/qr.cpp
/// @brief Sequential Householder QR factorization.
///
/// See linalg/factorization/qr.cpp for the full algorithm commentary.
#include "impl.hpp"
#include <cmath>
#include <vector>

namespace num::backends::seq {

QRResult qr(const Matrix& A) {
    constexpr real householder_tol = 1e-14;
    const idx m = A.rows();
    const idx n = A.cols();
    const idx r = (m > n) ? n : m - 1;

    Matrix R = A;

    std::vector<std::vector<real>> vs(r);

    for (idx k = 0; k < r; ++k) {
        const idx len = m - k;

        std::vector<real> x(len);
        for (idx i = 0; i < len; ++i) x[i] = R(k + i, k);

        real norm_x = real(0);
        for (idx i = 0; i < len; ++i) norm_x += x[i] * x[i];
        norm_x = std::sqrt(norm_x);

        std::vector<real> v = x;
        v[0] += (x[0] >= real(0)) ? norm_x : -norm_x;

        real norm_v = real(0);
        for (idx i = 0; i < len; ++i) norm_v += v[i] * v[i];
        norm_v = std::sqrt(norm_v);

        if (norm_v < householder_tol) {
            vs[k].assign(len, real(0));
            continue;
        }
        for (idx i = 0; i < len; ++i) v[i] /= norm_v;
        vs[k] = v;

        for (idx j = k; j < n; ++j) {
            real vTr = real(0);
            for (idx i = 0; i < len; ++i) vTr += v[i] * R(k + i, j);
            const real two_vTr = real(2) * vTr;
            for (idx i = 0; i < len; ++i) R(k + i, j) -= two_vTr * v[i];
        }
    }

    Matrix Q(m, m, real(0));
    for (idx i = 0; i < m; ++i) Q(i, i) = real(1);

    for (idx k = r; k-- > 0; ) {
        const std::vector<real>& v = vs[k];
        const idx len = static_cast<idx>(v.size());

        for (idx j = k; j < m; ++j) {
            real vTq = real(0);
            for (idx i = 0; i < len; ++i) vTq += v[i] * Q(k + i, j);
            const real two_vTq = real(2) * vTq;
            for (idx i = 0; i < len; ++i) Q(k + i, j) -= two_vTq * v[i];
        }
    }

    for (idx i = 1; i < m; ++i)
        for (idx j = 0; j < std::min(i, n); ++j)
            R(i, j) = real(0);

    return {std::move(Q), std::move(R)};
}

} // namespace num::backends::seq
