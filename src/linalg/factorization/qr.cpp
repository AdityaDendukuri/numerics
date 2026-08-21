/// @file linalg/factorization/qr.cpp
/// @brief QR factorization dispatcher + implementations (sequential & LAPACK).

#include "linalg/factorization/qr.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(NUMERICS_HAS_LAPACK)
  #include "core/parallel/lapack_wrapper.hpp"
#endif

namespace num {

namespace backends {

namespace seq {
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
    for (idx i = 0; i < len; ++i) {
      x[i] = R(k + i, k);
    }

    real norm_x = real(0);
    for (idx i = 0; i < len; ++i) {
      norm_x += x[i] * x[i];
    }
    norm_x = std::sqrt(norm_x);

    std::vector<real> v = x;
    v[0] += (x[0] >= real(0)) ? norm_x : -norm_x;

    real norm_v = real(0);
    for (idx i = 0; i < len; ++i) {
      norm_v += v[i] * v[i];
    }
    norm_v = std::sqrt(norm_v);

    if (norm_v < householder_tol) {
      vs[k].assign(len, real(0));
      continue;
    }
    for (idx i = 0; i < len; ++i) {
      v[i] /= norm_v;
    }
    vs[k] = v;

    for (idx j = k; j < n; ++j) {
      real vTr = real(0);
      for (idx i = 0; i < len; ++i) {
        vTr += v[i] * R(k + i, j);
      }
      const real two_vTr = real(2) * vTr;
      for (idx i = 0; i < len; ++i) {
        R(k + i, j) -= two_vTr * v[i];
      }
    }
  }

  Matrix Q(m, m, real(0));
  for (idx i = 0; i < m; ++i) {
    Q(i, i) = real(1);
  }

  for (idx k = r; k-- > 0;) {
    const std::vector<real>& v = vs[k];
    const idx len = static_cast<idx>(v.size());

    for (idx j = k; j < m; ++j) {
      real vTq = real(0);
      for (idx i = 0; i < len; ++i) {
        vTq += v[i] * Q(k + i, j);
      }
      const real two_vTq = real(2) * vTq;
      for (idx i = 0; i < len; ++i) {
        Q(k + i, j) -= two_vTq * v[i];
      }
    }
  }

  for (idx i = 1; i < m; ++i) {
    for (idx j = 0; j < std::min(i, n); ++j) {
      R(i, j) = real(0);
    }
  }

  return {std::move(Q), std::move(R)};
}
} // namespace seq

namespace lapack {
QRResult qr(const Matrix& A) {
#if defined(NUMERICS_HAS_LAPACK)
  const idx m = A.rows(), n = A.cols();
  const idx k = std::min(m, n);

  Matrix R = A;
  std::vector<double> tau(k);

  int info = LAPACKE_dgeqrf(LAPACK_ROW_MAJOR,
                            static_cast<lapack_int>(m),
                            static_cast<lapack_int>(n),
                            R.data(),
                            static_cast<lapack_int>(n),
                            tau.data());
  if (info != 0) {
    throw std::runtime_error("qr (lapack): dgeqrf failed, info=" + std::to_string(info));
  }

  Matrix Rmat = R;
  for (idx i = 1; i < m; ++i) {
    for (idx j = 0; j < std::min(i, n); ++j) {
      Rmat(i, j) = 0.0;
    }
  }

  Matrix Q(m, m, 0.0);
  for (idx j = 0; j < k; ++j) {
    for (idx i = 0; i < m; ++i) {
      Q(i, j) = R(i, j);
    }
  }

  info = LAPACKE_dorgqr(LAPACK_ROW_MAJOR,
                        static_cast<lapack_int>(m),
                        static_cast<lapack_int>(m),
                        static_cast<lapack_int>(k),
                        Q.data(),
                        static_cast<lapack_int>(m),
                        tau.data());
  if (info != 0) {
    throw std::runtime_error("qr (lapack): dorgqr failed, info=" + std::to_string(info));
  }

  return {std::move(Q), std::move(Rmat)};
#else
  return seq::qr(A);
#endif
}
} // namespace lapack

} // namespace backends

QRResult qr(const Matrix& A, Backend backend) {
  switch (backend) {
    case Backend::lapack:
      return backends::lapack::qr(A);
    default:
      return backends::seq::qr(A);
  }
}

void qr_solve(const QRResult& f, const Vector& b, Vector& x) {
  const idx m = f.Q.rows();
  const idx n = f.R.cols();

  Vector y(m, real(0));
  for (idx i = 0; i < m; ++i) {
    for (idx j = 0; j < m; ++j) {
      y[i] += f.Q(j, i) * b[j];
    }
  }

  Vector xv(n, real(0));
  for (idx i = n; i-- > 0;) {
    xv[i] = y[i];
    for (idx j = i + 1; j < n; ++j) {
      xv[i] -= f.R(i, j) * xv[j];
    }
    xv[i] /= f.R(i, i);
  }

  x = std::move(xv);
}

} // namespace num
