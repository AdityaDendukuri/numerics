/// @file linalg/factorization/backends/lapack/thomas.cpp
/// @brief LAPACK tridiagonal solver via LAPACKE_dgtsv.
#include "../seq/impl.hpp"
#include "impl.hpp"
#include <stdexcept>
#include <string>
#include <vector>

#if defined(NUMERICS_HAS_LAPACK)
#include <lapacke.h>
#endif

namespace num::backends::lapack {

void thomas(const Vector &a, const Vector &b, const Vector &c, const Vector &d,
            Vector &x) {
#if defined(NUMERICS_HAS_LAPACK)
  const idx n = b.size();
  // dgtsv overwrites its inputs; work on copies
  std::vector<double> dl(a.data(), a.data() + (n - 1));
  std::vector<double> diag(b.data(), b.data() + n);
  std::vector<double> du(c.data(), c.data() + (n - 1));
  x = d;
  int info = LAPACKE_dgtsv(LAPACK_ROW_MAJOR, static_cast<lapack_int>(n), 1,
                           dl.data(), diag.data(), du.data(), x.data(),
                           1); // ldb = nrhs = 1 (row-major: cols of RHS)
  if (info != 0)
    throw std::runtime_error("thomas (lapack): dgtsv failed, info=" +
                             std::to_string(info));
#else
  seq::thomas(a, b, c, d, x);
#endif
}

} // namespace num::backends::lapack
