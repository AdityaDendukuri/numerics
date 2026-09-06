/// @file banded.hpp
/// @brief banded matrix storage and solvers.
#pragma once

#include "kernel/factor.hpp"
#include "kernel/kernel.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "cuda/cuda_ops.hpp"

#include "core/policy.hpp"
#include "core/types.hpp"
#include "container/vector.hpp"
#include <memory>

namespace num {

/// @brief LAPACK-style band storage.
///
/// Stores \f$A_{ij}\f$ at \f$\text{band}(k_l+k_u+i-j,j)\f$ when
/// \f$\max(0,j-k_u)\le i\le \min(n-1,j+k_l)\f$.
class band_mat {
  public:
    /// Construct a zero-filled n-by-n matrix with lower/upper bandwidths kl/ku.
    band_mat(idx n, idx kl, idx ku);

    /// Construct a banded matrix with every stored entry initialized to val.
    band_mat(idx n, idx kl, idx ku, real val);

    ~band_mat();

    band_mat(const band_mat &);
    band_mat(band_mat &&) noexcept;
    band_mat &operator=(const band_mat &);
    band_mat &operator=(band_mat &&) noexcept;

    /// Return the square matrix order.
    [[nodiscard]] idx size() const { return n_; }
    [[nodiscard]] idx rows() const { return n_; }
    [[nodiscard]] idx cols() const { return n_; }

    /// Return the lower bandwidth.
    [[nodiscard]] idx kl() const { return kl_; }

    /// Return the upper bandwidth.
    [[nodiscard]] idx ku() const { return ku_; }

    /// Return the number of mathematical diagonals in the band.
    [[nodiscard]] idx bandwidth() const { return kl_ + ku_ + 1; }

    /// Return the leading dimension of the LAPACK-compatible storage.
    [[nodiscard]] idx ldab() const { return ldab_; }

    /// Access a mathematical matrix entry inside the stored band.
    real &operator()(idx i, idx j);
    real operator()(idx i, idx j) const;

    /// Access an entry by physical band-storage coordinates.
    real &band(idx band_row, idx col);
    [[nodiscard]] real band(idx band_row, idx col) const;

    real *data() { return data_.get(); }
    [[nodiscard]] const real *data() const { return data_.get(); }

    /// Test whether mathematical entry (i,j) is explicitly stored.
    [[nodiscard]] bool in_band(idx i, idx j) const;

    void to_gpu();
    void to_cpu();
    real *gpu_data() { return d_data_; }
    [[nodiscard]] const real *gpu_data() const { return d_data_; }
    [[nodiscard]] bool on_gpu() const { return d_data_ != nullptr; }

  private:
    idx n_ = 0;
    idx kl_ = 0;
    idx ku_ = 0;
    idx ldab_ = 0;
    std::unique_ptr<real[]> data_;
    real *d_data_ = nullptr;
};

#include <ostream>

/// Status and diagnostics from a banded factorization or solve.
struct banded_solver_result {
    bool success = false;
    idx pivot_row = 0;
    real rcond = 0.0;

    friend std::ostream &operator<<(std::ostream &os, const banded_solver_result &r) {
        os << "banded_solver_result{ success: " << (r.success ? "true" : "false")
           << ", rcond: " << r.rcond << " }";
        return os;
    }
};

/// @brief In-place banded \f$PA=LU\f$ factorization.
banded_solver_result banded_lu(band_mat &A, idx *ipiv);

/// @brief Solve \f$Ax=b\f$ using a precomputed banded LU factorization.
void banded_lu_solve(const band_mat &A, const idx *ipiv, vec &b);

/// @brief Solve \f$AX=B\f$ using a precomputed banded LU factorization.
void banded_lu_solve_multi(const band_mat &A, const idx *ipiv, real *B, idx nrhs);

/// @brief Factor and solve \f$Ax=b\f$.
banded_solver_result banded_solve(const band_mat &A, const vec &b, vec &x);

/// @brief Compute \f$y=Ax\f$.
void banded_matvec(const band_mat &A, const vec &x, vec &y);

/// @brief Compute \f$y=\alpha Ax+\beta y\f$.
void banded_gemv(real alpha, const band_mat &A, const vec &x, real beta, vec &y);

/// @brief Estimate \f$1/\kappa_1(A)\f$.
real banded_rcond(const band_mat &A, const idx *ipiv, real anorm);

/// @brief Compute \f$\|A\|_1\f$.
real banded_norm1(const band_mat &A);

inline band_mat::band_mat(idx n, idx kl, idx ku)
    : n_(n), kl_(kl), ku_(ku), ldab_((2 * kl) + ku + 1) {
    if (n == 0) {
        throw std::invalid_argument("band_mat: n must be positive");
    }
    data_ = std::make_unique<real[]>(ldab_ * n_);
}

inline band_mat::band_mat(idx n, idx kl, idx ku, real val) : band_mat(n, kl, ku) {
    std::fill_n(data_.get(), ldab_ * n_, val);
}

inline band_mat::~band_mat() {
#ifdef NUMERICS_HAS_CUDA
    if (d_data_)
        cuda::free(d_data_);
#endif
}

inline band_mat::band_mat(const band_mat &other)
    : n_(other.n_), kl_(other.kl_), ku_(other.ku_), ldab_(other.ldab_) {
    data_ = std::make_unique<real[]>(ldab_ * n_);
    std::memcpy(data_.get(), other.data_.get(), ldab_ * n_ * sizeof(real));
}

inline band_mat::band_mat(band_mat &&other) noexcept
    : n_(other.n_), kl_(other.kl_), ku_(other.ku_), ldab_(other.ldab_),
      data_(std::move(other.data_)), d_data_(other.d_data_) {
    other.n_ = 0;
    other.d_data_ = nullptr;
}

inline band_mat &band_mat::operator=(const band_mat &other) {
    if (this != &other) {
        n_ = other.n_;
        kl_ = other.kl_;
        ku_ = other.ku_;
        ldab_ = other.ldab_;
        data_ = std::make_unique<real[]>(ldab_ * n_);
        std::memcpy(data_.get(), other.data_.get(), ldab_ * n_ * sizeof(real));
#ifdef NUMERICS_HAS_CUDA
        if (d_data_) {
            cuda::free(d_data_);
            d_data_ = nullptr;
        }
#endif
    }
    return *this;
}

inline band_mat &band_mat::operator=(band_mat &&other) noexcept {
    if (this != &other) {
#ifdef NUMERICS_HAS_CUDA
        if (d_data_)
            cuda::free(d_data_);
#endif
        n_ = other.n_;
        kl_ = other.kl_;
        ku_ = other.ku_;
        ldab_ = other.ldab_;
        data_ = std::move(other.data_);
        d_data_ = other.d_data_;
        other.n_ = 0;
        other.d_data_ = nullptr;
    }
    return *this;
}

inline real &band_mat::operator()(idx i, idx j) {
    return data_[(kl_ + ku_ + i - j) + (j * ldab_)];
}

inline real band_mat::operator()(idx i, idx j) const {
    return data_[(kl_ + ku_ + i - j) + (j * ldab_)];
}

inline real &band_mat::band(idx band_row, idx col) {
    return data_[band_row + (col * ldab_)];
}

inline real band_mat::band(idx band_row, idx col) const {
    return data_[band_row + (col * ldab_)];
}

inline bool band_mat::in_band(idx i, idx j) const {
    return (j <= i + ku_) && (i <= j + kl_);
}

inline void band_mat::to_gpu() {
#ifdef NUMERICS_HAS_CUDA
    if (!d_data_)
        d_data_ = cuda::alloc(ldab_ * n_);
    cuda::to_device(d_data_, data_.get(), ldab_ * n_);
#endif
}

inline void band_mat::to_cpu() {
#ifdef NUMERICS_HAS_CUDA
    if (d_data_)
        cuda::to_host(data_.get(), d_data_, ldab_ * n_);
#endif
}

// LU Factorization with Partial Pivoting

inline banded_solver_result banded_lu(band_mat &A, idx *ipiv) {
    const idx n = A.size(), kl = A.kl(), ku = A.ku(), ldab = A.ldab();
    real *ab = A.data();
    banded_solver_result result{true, 0, 0.0};

    const bool ok = kernel::banded_factor(ab, ldab, n, kl, ku, ipiv);
    if (!ok) {
        result.success = false;
        return result;
    }
    return result;
}

// Solve Using LU Factorization

inline void banded_lu_solve(const band_mat &A, const idx *ipiv, vec &b) {
    const idx n = A.size(), kl = A.kl(), ku = A.ku(), ldab = A.ldab();
    const real *ab = A.data();
    real *x = b.data();
    if (b.size() != n) {
        throw std::invalid_argument("banded_lu_solve: dimension mismatch");
    }

    kernel::banded_solve(x, ab, ldab, n, kl, ku, ipiv);
}

inline void banded_lu_solve_multi(const band_mat &A, const idx *ipiv, real *B, idx nrhs) {
    const idx n = A.size(), kl = A.kl(), ku = A.ku(), ldab = A.ldab();
    const real *ab = A.data();
    const idx kv = ku + kl;

#ifdef _OPENMP
#pragma omp parallel for if (nrhs > 16)
#endif
    for (idx rhs = 0; rhs < nrhs; ++rhs) {
        real *x = B + (rhs * n);
        for (idx i = 0; i < n; ++i) {
            if (ipiv[i] != i) {
                std::swap(x[i], x[ipiv[i]]);
            }
        }
        for (idx j = 0; j < n; ++j) {
            if (x[j] != 0.0) {
                const idx last = std::min(j + kl, n - 1);
                real xj = x[j];
                for (idx i = j + 1; i <= last; ++i) {
                    x[i] -= ab[kv + i - j + (j * ldab)] * xj;
                }
            }
        }
        for (idx j = n; j > 0; --j) {
            const idx col = j - 1;
            x[col] /= ab[kv + (col * ldab)];
            if (x[col] != 0.0) {
                const idx first = (col > ku) ? col - ku : 0;
                real xc = x[col];
                for (idx i = first; i < col; ++i) {
                    x[i] -= ab[kv + i - col + (col * ldab)] * xc;
                }
            }
        }
    }
}

inline banded_solver_result banded_solve(const band_mat &A, const vec &b, vec &x) {
    const idx n = A.size();
    if (b.size() != n || x.size() != n) {
        throw std::invalid_argument("banded_solve: dimension mismatch");
    }

    band_mat a_work = A;
    auto ipiv = std::make_unique<idx[]>(n);
    banded_solver_result result = banded_lu(a_work, ipiv.get());
    if (!result.success) {
        return result;
    }

    for (idx i = 0; i < n; ++i) {
        x[i] = b[i];
    }
    banded_lu_solve(a_work, ipiv.get(), x);
    return result;
}

// mat-vec Products

inline void banded_matvec(const band_mat &A, const vec &x, vec &y) {
    banded_gemv(1.0, A, x, 0.0, y);
}

inline void banded_gemv(real alpha, const band_mat &A, const vec &x, real beta, vec &y) {
    const idx n = A.size(), kl = A.kl(), ku = A.ku();
    if (x.size() != n || y.size() != n) {
        throw std::invalid_argument("banded_gemv: dimension mismatch");
    }

    const idx ldab = A.ldab();
    const real *ab = A.data();
    const real *xp = x.data();
    real *yp = y.data();

    kernel::gbmv(yp, alpha, ab, ldab, kl, ku, xp, beta, n);
}

// Condition Number Estimation

inline real banded_norm1(const band_mat &A) {
    const idx n = A.size(), kl = A.kl(), ku = A.ku(), ldab = A.ldab();
    const real *ab = A.data();
    const idx kv = ku + kl;
    real max_sum = 0.0;
    for (idx j = 0; j < n; ++j) {
        real col_sum = 0.0;
        const idx i_start = (j > ku) ? j - ku : 0;
        const idx i_end = std::min(j + kl, n - 1);
        for (idx i = i_start; i <= i_end; ++i) {
            col_sum += std::abs(ab[kv + i - j + (j * ldab)]);
        }
        max_sum = std::max(max_sum, col_sum);
    }
    return max_sum;
}

inline real banded_rcond(const band_mat &A, const idx *ipiv, real anorm) {
    const idx n = A.size();
    if (n == 0 || anorm == 0.0) {
        return 0.0;
    }
    vec y(n, 1.0 / static_cast<real>(n));
    const band_mat &a_copy = A;
    banded_lu_solve(a_copy, ipiv, y);
    real ainv_norm = 0.0;
    for (idx i = 0; i < n; ++i) {
        ainv_norm += std::abs(y[i]);
    }
    return 1.0 / (anorm * ainv_norm);
}

} // namespace num
