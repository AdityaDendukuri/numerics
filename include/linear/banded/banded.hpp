/// @file banded.hpp
/// @brief banded matrix storage and solvers.
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
#include <memory>

namespace num {

/// @brief banded LU factorization with partial pivoting over LAPACK-compatible band storage,
/// as LAPACK `gbtf2`.
///
/// `ab` is column-major with `ldab >= 2*kl + ku + 1` rows; \f$A_{ij}\f$ sits at band row
/// `kl + ku + i - j`. A row swap can bring in entries up to `kl + ku` past the diagonal, so
/// \f$U\f$ has bandwidth `kl + ku` and fills the first `kl` band rows, which are zeroed
/// here. The multipliers of \f$L\f$ stay where each column's elimination wrote them: later
/// swaps are not applied to them, so `banded_solve` interleaves the swaps with the forward
/// substitution. Returns false if a pivot column is entirely zero.
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

/// @brief Solve \f$Ax = b\f$ in place from `banded_factor`, as LAPACK `gbtrs`.
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

/// @brief The banded factorization \f$PA = LU\f$.
///
/// `LU` holds both factors in band storage: \f$U\f$, with the fill pivoting adds, in the
/// rows above the diagonal row, and the multipliers of \f$L\f$ below it. \f$L\f$ has a unit
/// diagonal, which is not stored. Step `k` exchanged rows `k` and `swaps[k]`.
struct banded_lu_result {
    band_mat LU;
    array<idx> swaps;
    bool singular = false; ///< True when a pivot column was entirely zero.

    [[nodiscard]] idx size() const { return LU.size(); }
};

/// @brief Factor \f$PA = LU\f$ with partial pivoting. Pass an rvalue to factor in place.
banded_lu_result lu(band_mat A);

/// @brief Solve \f$Ax = b\f$. `x` may be `b`.
void solve(const banded_lu_result &f, const vec<real> &b, vec<real> &x);

/// @brief Solve \f$AX = B\f$, one right-hand side per column. `X` may be `B`.
void solve(const banded_lu_result &f, const mat<real> &B, mat<real> &X);

/// @brief Compute \f$y=Ax\f$.
void banded_matvec(const band_mat &A, const vec<real> &x, vec<real> &y);

/// @brief Compute \f$y=\alpha Ax+\beta y\f$.
void banded_gemv(real alpha, const band_mat &A, const vec<real> &x, real beta, vec<real> &y);

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

inline banded_lu_result lu(band_mat A) {
    const idx n = A.size();
    banded_lu_result f{std::move(A), array<idx>(n), false};
    f.singular =
        !num::banded_factor(f.LU.data(), f.LU.ldab(), n, f.LU.kl(), f.LU.ku(), f.swaps.data());
    return f;
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

} // namespace num
