/// @file linear/banded/banded.hpp
/// @brief Banded matrices and solvers.
#pragma once

#include "kernel/kernel.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include "cuda/cuda_ops.hpp"

#include "core/policy.hpp"
#include "core/types.hpp"
#include "container/matrix.hpp"
#include "container/vector.hpp"
#include "blas/vector_ops.hpp"
#include "lapack/lapack_wrapper.hpp"
#include <memory>

namespace num {

/// @brief Banded LU factorization, partial pivoting (GBTF2): \f$P A = L U\f$ in band storage.
/// Needs `ldab >= 2 kl + ku + 1`. Returns false on a zero pivot column.
template <std::floating_point T, class Index>
[[nodiscard]] inline bool banded_factor(T *NUM_K_RESTRICT ab, idx ldab, idx n, idx kl, idx ku,
                                        Index *NUM_K_RESTRICT ipiv) noexcept {
    const idx kv = ku + kl;
    auto at = [&](idx i, idx j) -> T & { return ab[kv + i - j + (j * ldab)]; };
    // Fill rows of column c hold rows above c - ku, which only columns past ku have. Zero
    // each one as the elimination first reaches it, as `gbtf2` does.
    auto zero_fill = [&](idx c) {
        for (idx r = 0; r < kl; ++r) {
            ab[r + (c * ldab)] = T(0);
        }
    };
    for (idx c = ku + 1; c < std::min(kv, n); ++c) {
        zero_fill(c);
    }
    idx ju = 0; // Last column the factored rows reach so far.
    for (idx j = 0; j < n; ++j) {
        if (j + kv < n) {
            zero_fill(j + kv);
        }
        const idx km = std::min(kl, n - 1 - j);
        idx pivot = j;
        T max_val = std::abs(at(j, j));
        for (idx i = j + 1; i <= j + km; ++i) {
            if (std::abs(at(i, j)) > max_val) {
                max_val = std::abs(at(i, j));
                pivot = i;
            }
        }
        ipiv[j] = static_cast<Index>(pivot);
        if (max_val == T(0)) {
            return false;
        }
        ju = std::max(ju, std::min(pivot + ku, n - 1));
        if (pivot != j) {
            for (idx c = j; c <= ju; ++c) {
                std::swap(at(j, c), at(pivot, c));
            }
        }
        const T inv_pivot = T(1) / at(j, j);
        for (idx i = j + 1; i <= j + km; ++i) {
            at(i, j) *= inv_pivot;
        }
        for (idx c = j + 1; c <= ju; ++c) {
            const T ujc = at(j, c);
            if (ujc != T(0)) {
                for (idx i = j + 1; i <= j + km; ++i) {
                    at(i, c) -= at(i, j) * ujc;
                }
            }
        }
    }
    return true;
}

/// @brief Banded LU solve (GBTRS), in place: \f$x \leftarrow A^{-1} b\f$.
template <std::floating_point T, class Index>
inline void banded_solve(T *x, const T *NUM_K_RESTRICT ab, idx ldab, idx n, idx kl, idx ku,
                         const Index *NUM_K_RESTRICT ipiv) noexcept {
    const idx kv = ku + kl;
    for (idx j = 0; j < n; ++j) {
        const idx p = static_cast<idx>(ipiv[j]);
        if (p != j) {
            std::swap(x[j], x[p]);
        }
        const T xj = x[j];
        if (xj != T(0)) {
            const idx last = std::min(j + kl, n - 1);
            for (idx i = j + 1; i <= last; ++i) {
                x[i] -= ab[kv + i - j + (j * ldab)] * xj;
            }
        }
    }
    for (idx col = n; col-- > 0;) {
        x[col] /= ab[kv + (col * ldab)];
        const T xc = x[col];
        if (xc != T(0)) {
            const idx first = (col > kv) ? col - kv : 0;
            for (idx i = first; i < col; ++i) {
                x[i] -= ab[kv + i - col + (col * ldab)] * xc;
            }
        }
    }
}

} // namespace num

namespace num {

/// @brief Banded matrix in LAPACK band storage: \f$a_{ij}\f$ at band row \f$k_l + k_u + i - j\f$.
class band_mat {
  public:
    /// @brief Zero \f$n \times n\f$ band matrix with bandwidths \f$k_l, k_u\f$.
    band_mat(idx n, idx kl, idx ku);

    /// @brief Constant \f$n \times n\f$ band matrix with bandwidths \f$k_l, k_u\f$.
    band_mat(idx n, idx kl, idx ku, real val);

    ~band_mat();

    band_mat(const band_mat &);
    band_mat(band_mat &&) noexcept;
    band_mat &operator=(const band_mat &);
    band_mat &operator=(band_mat &&) noexcept;

    /// @brief Order \f$n\f$.
    [[nodiscard]] idx size() const { return n_; }
    [[nodiscard]] idx rows() const { return n_; }
    [[nodiscard]] idx cols() const { return n_; }

    /// @brief Lower bandwidth \f$k_l\f$.
    [[nodiscard]] idx kl() const { return kl_; }

    /// @brief Upper bandwidth \f$k_u\f$.
    [[nodiscard]] idx ku() const { return ku_; }

    /// @brief Number of diagonals: \f$k_l + k_u + 1\f$.
    [[nodiscard]] idx bandwidth() const { return kl_ + ku_ + 1; }

    /// @brief Leading dimension of the band storage.
    [[nodiscard]] idx ldab() const { return ldab_; }

    /// @brief Entry \f$a_{ij}\f$ inside the band.
    real &operator()(idx i, idx j);
    real operator()(idx i, idx j) const;

    /// @brief Entry by band-storage coordinates.
    real &band(idx band_row, idx col);
    [[nodiscard]] real band(idx band_row, idx col) const;

    real *data() { return data_.get(); }
    [[nodiscard]] const real *data() const { return data_.get(); }

    /// @brief Whether \f$a_{ij}\f$ is inside the band.
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

/// @brief Banded LU result: \f$P A = L U\f$ in band storage.
struct banded_lu_result {
    band_mat LU;
    array<idx> swaps;
    bool singular = false; ///< True when a pivot column was entirely zero.

    [[nodiscard]] idx size() const { return LU.size(); }
};

/// @brief Banded LU factorization, partial pivoting: \f$P A = L U\f$.
/// Pass an rvalue to factor in place.
banded_lu_result lu(band_mat A);

/// @brief Banded LU solve: \f$x \leftarrow A^{-1} b\f$. `x` may alias `b`.
void solve(const banded_lu_result &f, const vec<real> &b, vec<real> &x);

/// @brief Banded LU solve: \f$X \leftarrow A^{-1} B\f$. `X` may alias `B`.
void solve(const banded_lu_result &f, const mat<real> &B, mat<real> &X);

/// @brief Banded matrix-vector product: \f$y \leftarrow A x\f$.
void banded_matvec(const band_mat &A, const vec<real> &x, vec<real> &y);

/// @brief Banded matrix-vector product (GBMV): \f$y \leftarrow \alpha A x + \beta y\f$.
void banded_gemv(real alpha, const band_mat &A, const vec<real> &x, real beta, vec<real> &y);

/// @brief Banded 1-norm: \f$\|A\|_1\f$.
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

namespace seq {

/// @brief Banded LU factorization, partial pivoting, in-tree: \f$P A = L U\f$.
inline banded_lu_result lu(band_mat A) {
    const idx n = A.size();
    banded_lu_result f{std::move(A), array<idx>(n), false};
    f.singular =
        !num::banded_factor(f.LU.data(), f.LU.ldab(), n, f.LU.kl(), f.LU.ku(), f.swaps.data());
    return f;
}

/// @brief Banded matrix-vector product (GBMV), in-tree: \f$y \leftarrow \alpha A x + \beta y\f$.
inline void banded_gemv(real alpha, const band_mat &A, const vec<real> &x, real beta, vec<real> &y) {
    if (x.size() != A.size() || y.size() != A.size()) {
        throw std::invalid_argument("banded_gemv: dimension mismatch");
    }
    kernel::gbmv(y.data(), alpha, A.data(), A.ldab(), A.kl(), A.ku(), x.data(), beta, A.size());
}

} // namespace seq

namespace blas {

/// @brief Banded matrix-vector product through BLAS `dgbmv`: \f$y \leftarrow \alpha A x + \beta y\f$.
inline void banded_gemv(real alpha, const band_mat &A, const vec<real> &x, real beta, vec<real> &y) {
    if (x.size() != A.size() || y.size() != A.size()) {
        throw std::invalid_argument("banded_gemv: dimension mismatch");
    }
#if defined(NUMERICS_HAS_BLAS)
    // The top kl rows of the storage hold LU fill; dgbmv's band starts below them.
    const int n = static_cast<int>(A.size());
    cblas_dgbmv(CblasColMajor, CblasNoTrans, n, n, static_cast<int>(A.kl()),
                static_cast<int>(A.ku()), alpha, A.data() + A.kl(), static_cast<int>(A.ldab()),
                x.data(), 1, beta, y.data(), 1);
#else
    warn_unavailable();
    seq::banded_gemv(alpha, A, x, beta, y);
#endif
}

} // namespace blas

namespace lapack {

/// @brief Banded LU factorization through LAPACK `dgbtrf`: \f$P A = L U\f$.
inline banded_lu_result lu(band_mat A) {
#if defined(NUMERICS_HAS_LAPACK)
    const idx n = A.size();
    banded_lu_result f{std::move(A), array<idx>(n), false};
    array<lapack_int> ipiv(n);
    const lapack_int info = LAPACKE_dgbtrf(
        LAPACK_COL_MAJOR, static_cast<lapack_int>(n), static_cast<lapack_int>(n),
        static_cast<lapack_int>(f.LU.kl()), static_cast<lapack_int>(f.LU.ku()), f.LU.data(),
        static_cast<lapack_int>(f.LU.ldab()), ipiv.data());
    if (info < 0) {
        throw std::runtime_error("banded lu (lapack): dgbtrf argument error");
    }
    f.singular = info > 0;
    for (idx k = 0; k < n; ++k) {
        f.swaps[k] = static_cast<idx>(ipiv[k] - 1);
    }
    return f;
#else
    return seq::lu(std::move(A));
#endif
}

} // namespace lapack

/// @brief Banded LU factorization, partial pivoting: \f$P A = L U\f$.
/// Uses LAPACK when an optimized LAPACK is available.
inline banded_lu_result lu(band_mat A) {
#if defined(NUMERICS_LAPACK_DEFAULT)
    return lapack::lu(std::move(A));
#else
    return seq::lu(std::move(A));
#endif
}

inline void solve(const banded_lu_result &f, const vec<real> &b, vec<real> &x) {
    if (b.size() != f.size()) {
        throw std::invalid_argument("banded solve: dimension mismatch");
    }
    x = b;
    num::banded_solve(x.data(), f.LU.data(), f.LU.ldab(), f.size(), f.LU.kl(), f.LU.ku(),
                      f.swaps.data());
}

inline void solve(const banded_lu_result &f, const mat<real> &B, mat<real> &X) {
    const idx n = f.size(), kl = f.LU.kl(), ku = f.LU.ku(), ldab = f.LU.ldab();
    if (B.rows() != n) {
        throw std::invalid_argument("banded solve: dimension mismatch");
    }
    X = B;
    const idx nrhs = X.cols();
    const idx kv = ku + kl;
    const real *ab = f.LU.data();
    real *x = X.data();
    // The vector solve, with each scalar update applied across a contiguous row of X.
    for (idx j = 0; j < n; ++j) {
        if (f.swaps[j] != j) {
            kernel::swap_rows(x, nrhs, j, f.swaps[j], nrhs);
        }
        const real *xj = x + (j * nrhs);
        const idx last = std::min(j + kl, n - 1);
        for (idx i = j + 1; i <= last; ++i) {
            const real l = ab[kv + i - j + (j * ldab)];
            real *xi = x + (i * nrhs);
            for (idx c = 0; c < nrhs; ++c) {
                xi[c] -= l * xj[c];
            }
        }
    }
    for (idx col = n; col-- > 0;) {
        real *xc = x + (col * nrhs);
        const real inverse_pivot = 1.0 / ab[kv + (col * ldab)];
        for (idx c = 0; c < nrhs; ++c) {
            xc[c] *= inverse_pivot;
        }
        const idx first = (col > kv) ? col - kv : 0;
        for (idx i = first; i < col; ++i) {
            const real u = ab[kv + i - col + (col * ldab)];
            real *xi = x + (i * nrhs);
            for (idx c = 0; c < nrhs; ++c) {
                xi[c] -= u * xc[c];
            }
        }
    }
}

// mat-vec Products

inline void banded_matvec(const band_mat &A, const vec<real> &x, vec<real> &y) {
    banded_gemv(1.0, A, x, 0.0, y);
}

inline void banded_gemv(real alpha, const band_mat &A, const vec<real> &x, real beta, vec<real> &y) {
#if defined(NUMERICS_HAS_BLAS)
    blas::banded_gemv(alpha, A, x, beta, y);
#else
    seq::banded_gemv(alpha, A, x, beta, y);
#endif
}

// Norm

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

namespace lapack {

#if defined(NUMERICS_HAS_LAPACK)
namespace detail {
inline array<lapack_int> banded_pivots(const banded_lu_result &f) {
    array<lapack_int> ipiv(f.size());
    for (idx k = 0; k < f.size(); ++k) {
        ipiv[k] = static_cast<lapack_int>(f.swaps[k] + 1);
    }
    return ipiv;
}
} // namespace detail
#endif

/// @brief Banded LU solve through LAPACK `dgbtrs`: \f$x \leftarrow A^{-1} b\f$. `x` may alias `b`.
inline void solve(const banded_lu_result &f, const vec<real> &b, vec<real> &x) {
#if defined(NUMERICS_HAS_LAPACK)
    if (b.size() != f.size()) {
        throw std::invalid_argument("banded solve: dimension mismatch");
    }
    if (&x != &b) {
        x = b;
    }
    const auto ipiv = detail::banded_pivots(f);
    const auto n = static_cast<lapack_int>(f.size());
    LAPACKE_dgbtrs(LAPACK_COL_MAJOR, 'N', n, static_cast<lapack_int>(f.LU.kl()),
                   static_cast<lapack_int>(f.LU.ku()), 1, f.LU.data(),
                   static_cast<lapack_int>(f.LU.ldab()), ipiv.data(), x.data(), n);
#else
    num::solve(f, b, x);
#endif
}

/// @brief Banded LU solve through LAPACK `dgbtrs`: \f$X \leftarrow A^{-1} B\f$. `X` may alias `B`.
inline void solve(const banded_lu_result &f, const mat<real> &B, mat<real> &X) {
#if defined(NUMERICS_HAS_LAPACK)
    const idx n = f.size(), nrhs = B.cols();
    if (B.rows() != n) {
        throw std::invalid_argument("banded solve: dimension mismatch");
    }
    // dgbtrs takes the band and the right-hand sides in one layout: column-major.
    array<real> columns(n * nrhs);
    for (idx i = 0; i < n; ++i) {
        for (idx c = 0; c < nrhs; ++c) {
            columns[(c * n) + i] = B(i, c);
        }
    }
    const auto ipiv = detail::banded_pivots(f);
    LAPACKE_dgbtrs(LAPACK_COL_MAJOR, 'N', static_cast<lapack_int>(n),
                   static_cast<lapack_int>(f.LU.kl()), static_cast<lapack_int>(f.LU.ku()),
                   static_cast<lapack_int>(nrhs), f.LU.data(), static_cast<lapack_int>(f.LU.ldab()),
                   ipiv.data(), columns.data(), static_cast<lapack_int>(n));
    X = mat<real>(n, nrhs);
    for (idx i = 0; i < n; ++i) {
        for (idx c = 0; c < nrhs; ++c) {
            X(i, c) = columns[(c * n) + i];
        }
    }
#else
    num::solve(f, B, X);
#endif
}

} // namespace lapack

} // namespace num
