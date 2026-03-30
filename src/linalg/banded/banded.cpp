/// @file banded.cpp
/// @brief High-performance banded matrix solver implementation

#include "linalg/banded/banded.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef NUMERICS_HAS_CUDA
#include "core/parallel/cuda_ops.hpp"
#endif

namespace num {

BandedMatrix::BandedMatrix(idx n, idx kl, idx ku)
    : n_(n), kl_(kl), ku_(ku), ldab_(2 * kl + ku + 1) {
    if (n == 0) throw std::invalid_argument("BandedMatrix: n must be positive");
    data_ = std::make_unique<real[]>(ldab_ * n_);
}

BandedMatrix::BandedMatrix(idx n, idx kl, idx ku, real val)
    : BandedMatrix(n, kl, ku) {
    std::fill_n(data_.get(), ldab_ * n_, val);
}

BandedMatrix::~BandedMatrix() {
#ifdef NUMERICS_HAS_CUDA
    if (d_data_) cuda::free(d_data_);
#endif
}

BandedMatrix::BandedMatrix(const BandedMatrix& other)
    : n_(other.n_), kl_(other.kl_), ku_(other.ku_), ldab_(other.ldab_) {
    data_ = std::make_unique<real[]>(ldab_ * n_);
    std::memcpy(data_.get(), other.data_.get(), ldab_ * n_ * sizeof(real));
}


BandedMatrix::BandedMatrix(BandedMatrix&& other) noexcept
    : n_(other.n_), kl_(other.kl_), ku_(other.ku_), ldab_(other.ldab_),
      data_(std::move(other.data_)), d_data_(other.d_data_) {
    other.n_ = 0;
    other.d_data_ = nullptr;
}

BandedMatrix& BandedMatrix::operator=(const BandedMatrix& other) {
    if (this != &other) {
        n_ = other.n_; kl_ = other.kl_; ku_ = other.ku_; ldab_ = other.ldab_;
        data_ = std::make_unique<real[]>(ldab_ * n_);
        std::memcpy(data_.get(), other.data_.get(), ldab_ * n_ * sizeof(real));
#ifdef NUMERICS_HAS_CUDA
        if (d_data_) { cuda::free(d_data_); d_data_ = nullptr; }
#endif
    }
    return *this;
}

BandedMatrix& BandedMatrix::operator=(BandedMatrix&& other) noexcept {
    if (this != &other) {
#ifdef NUMERICS_HAS_CUDA
        if (d_data_) cuda::free(d_data_);
#endif
        n_ = other.n_; kl_ = other.kl_; ku_ = other.ku_; ldab_ = other.ldab_;
        data_ = std::move(other.data_);
        d_data_ = other.d_data_;
        other.n_ = 0; other.d_data_ = nullptr;
    }
    return *this;
}

real& BandedMatrix::operator()(idx i, idx j) {
    return data_[(kl_ + ku_ + i - j) + j * ldab_];
}

real BandedMatrix::operator()(idx i, idx j) const {
    return data_[(kl_ + ku_ + i - j) + j * ldab_];
}

real& BandedMatrix::band(idx band_row, idx col) {
    return data_[band_row + col * ldab_];
}

real BandedMatrix::band(idx band_row, idx col) const {
    return data_[band_row + col * ldab_];
}

bool BandedMatrix::in_band(idx i, idx j) const {
    return (j <= i + ku_) && (i <= j + kl_);
}

void BandedMatrix::to_gpu() {
#ifdef NUMERICS_HAS_CUDA
    if (!d_data_) d_data_ = cuda::alloc(ldab_ * n_);
    cuda::to_device(d_data_, data_.get(), ldab_ * n_);
#endif
}

void BandedMatrix::to_cpu() {
#ifdef NUMERICS_HAS_CUDA
    if (d_data_) cuda::to_host(data_.get(), d_data_, ldab_ * n_);
#endif
}

// LU Factorization with Partial Pivoting

BandedSolverResult banded_lu(BandedMatrix& A, idx* ipiv) {
    const idx n = A.size(), kl = A.kl(), ku = A.ku(), ldab = A.ldab();
    real* ab = A.data();
    BandedSolverResult result{true, 0, 0.0};

    for (idx i = 0; i < n; ++i) ipiv[i] = i;

    for (idx j = 0; j < n; ++j) {
        const idx kv = ku + kl;
        const idx last_row = std::min(j + kl, n - 1);

        real max_val = std::abs(ab[kv + j * ldab]);
        idx pivot = j;
        for (idx i = j + 1; i <= last_row; ++i) {
            real val = std::abs(ab[kv + i - j + j * ldab]);
            if (val > max_val) { max_val = val; pivot = i; }
        }

        if (max_val == 0.0) { result.success = false; result.pivot_row = j; return result; }

        ipiv[j] = pivot;
        if (pivot != j) {
            const idx col_start = (j > ku) ? j - ku : 0;
            const idx col_end = std::min(j + kl, n - 1);
            for (idx col = col_start; col <= col_end; ++col) {
                idx band_j = kl + ku + j - col;
                idx band_pivot = kl + ku + pivot - col;
                if (band_j < ldab && band_pivot < ldab)
                    std::swap(ab[band_j + col * ldab], ab[band_pivot + col * ldab]);
            }
        }

        real pivot_val = ab[kv + j * ldab];
        const idx num_rows = std::min(kl, n - j - 1);

        if (num_rows > 0 && pivot_val != 0.0) {
            real inv_pivot = 1.0 / pivot_val;
            #ifdef _OPENMP
            #pragma omp simd
            #endif
            for (idx i = 1; i <= num_rows; ++i)
                ab[kv + i + j * ldab] *= inv_pivot;

            const idx col_end = std::min(j + ku, n - 1);
            for (idx k = j + 1; k <= col_end; ++k) {
                real akj = ab[kv + j - k + k * ldab];
                if (akj != 0.0) {
                    #ifdef _OPENMP
                    #pragma omp simd
                    #endif
                    for (idx i = 1; i <= num_rows; ++i) {
                        idx band_row = kv + j + i - k;
                        if (band_row < ldab)
                            ab[band_row + k * ldab] -= ab[kv + i + j * ldab] * akj;
                    }
                }
            }
        }
    }
    return result;
}

// Solve Using LU Factorization

void banded_lu_solve(const BandedMatrix& A, const idx* ipiv, Vector& b) {
    const idx n = A.size(), kl = A.kl(), ku = A.ku(), ldab = A.ldab();
    const real* ab = A.data();
    real* x = b.data();
    if (b.size() != n) throw std::invalid_argument("banded_lu_solve: dimension mismatch");

    const idx kv = ku + kl;

    for (idx i = 0; i < n; ++i)
        if (ipiv[i] != i) std::swap(x[i], x[ipiv[i]]);

    for (idx j = 0; j < n; ++j) {
        if (x[j] != 0.0) {
            const idx last = std::min(j + kl, n - 1);
            real xj = x[j];
            #ifdef _OPENMP
            #pragma omp simd
            #endif
            for (idx i = j + 1; i <= last; ++i)
                x[i] -= ab[kv + i - j + j * ldab] * xj;
        }
    }

    for (idx j = n; j > 0; --j) {
        const idx col = j - 1;
        x[col] /= ab[kv + col * ldab];
        if (x[col] != 0.0) {
            const idx first = (col > ku) ? col - ku : 0;
            real xc = x[col];
            #ifdef _OPENMP
            #pragma omp simd
            #endif
            for (idx i = first; i < col; ++i)
                x[i] -= ab[kv + i - col + col * ldab] * xc;
        }
    }
}

void banded_lu_solve_multi(const BandedMatrix& A, const idx* ipiv,
                           real* B, idx nrhs) {
    const idx n = A.size(), kl = A.kl(), ku = A.ku(), ldab = A.ldab();
    const real* ab = A.data();
    const idx kv = ku + kl;

    #ifdef _OPENMP
    #pragma omp parallel for if(nrhs > 16)
    #endif
    for (idx rhs = 0; rhs < nrhs; ++rhs) {
        real* x = B + rhs * n;
        for (idx i = 0; i < n; ++i)
            if (ipiv[i] != i) std::swap(x[i], x[ipiv[i]]);
        for (idx j = 0; j < n; ++j) {
            if (x[j] != 0.0) {
                const idx last = std::min(j + kl, n - 1);
                real xj = x[j];
                for (idx i = j + 1; i <= last; ++i)
                    x[i] -= ab[kv + i - j + j * ldab] * xj;
            }
        }
        for (idx j = n; j > 0; --j) {
            const idx col = j - 1;
            x[col] /= ab[kv + col * ldab];
            if (x[col] != 0.0) {
                const idx first = (col > ku) ? col - ku : 0;
                real xc = x[col];
                for (idx i = first; i < col; ++i)
                    x[i] -= ab[kv + i - col + col * ldab] * xc;
            }
        }
    }
}

BandedSolverResult banded_solve(const BandedMatrix& A, const Vector& b, Vector& x) {
    const idx n = A.size();
    if (b.size() != n || x.size() != n)
        throw std::invalid_argument("banded_solve: dimension mismatch");

    BandedMatrix A_work = A;
    auto ipiv = std::make_unique<idx[]>(n);
    BandedSolverResult result = banded_lu(A_work, ipiv.get());
    if (!result.success) return result;

    for (idx i = 0; i < n; ++i) x[i] = b[i];
    banded_lu_solve(A_work, ipiv.get(), x);
    return result;
}

// Matrix-Vector Products

void banded_matvec(const BandedMatrix& A, const Vector& x, Vector& y, Backend backend) {
    banded_gemv(1.0, A, x, 0.0, y, backend);
}

void banded_gemv(real alpha, const BandedMatrix& A, const Vector& x,
                 real beta, Vector& y, Backend backend) {
    const idx n = A.size(), kl = A.kl(), ku = A.ku();
    if (x.size() != n || y.size() != n)
        throw std::invalid_argument("banded_gemv: dimension mismatch");

    if (backend != Backend::gpu) {
        const idx ldab = A.ldab();
        const real* ab = A.data();
        const real* xp = x.data();
        real* yp = y.data();
        const idx kv = ku + kl;

        if (beta == 0.0) std::memset(yp, 0, n * sizeof(real));
        else if (beta != 1.0) for (idx i = 0; i < n; ++i) yp[i] *= beta;

        #ifdef _OPENMP
        #pragma omp parallel for schedule(static) if(n > 1000)
        #endif
        for (idx j = 0; j < n; ++j) {
            if (xp[j] != 0.0) {
                real temp = alpha * xp[j];
                const idx i_start = (j > ku) ? j - ku : 0;
                const idx i_end = std::min(j + kl, n - 1);
                for (idx i = i_start; i <= i_end; ++i) {
                    #ifdef _OPENMP
                    #pragma omp atomic
                    #endif
                    yp[i] += ab[kv + i - j + j * ldab] * temp;
                }
            }
        }
    }
}

// Condition Number Estimation

real banded_norm1(const BandedMatrix& A) {
    const idx n = A.size(), kl = A.kl(), ku = A.ku(), ldab = A.ldab();
    const real* ab = A.data();
    const idx kv = ku + kl;
    real max_sum = 0.0;
    for (idx j = 0; j < n; ++j) {
        real col_sum = 0.0;
        const idx i_start = (j > ku) ? j - ku : 0;
        const idx i_end = std::min(j + kl, n - 1);
        for (idx i = i_start; i <= i_end; ++i)
            col_sum += std::abs(ab[kv + i - j + j * ldab]);
        max_sum = std::max(max_sum, col_sum);
    }
    return max_sum;
}

real banded_rcond(const BandedMatrix& A, const idx* ipiv, real anorm) {
    const idx n = A.size();
    if (n == 0 || anorm == 0.0) return 0.0;
    Vector y(n, 1.0 / static_cast<real>(n));
    BandedMatrix A_copy = A;
    banded_lu_solve(A_copy, ipiv, y);
    real ainv_norm = 0.0;
    for (idx i = 0; i < n; ++i) ainv_norm += std::abs(y[i]);
    return 1.0 / (anorm * ainv_norm);
}

} // namespace num
