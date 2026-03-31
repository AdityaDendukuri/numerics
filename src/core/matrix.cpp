/// @file core/matrix.cpp
/// @brief Matrix constructors, GPU lifecycle, and backend dispatch for matrix
/// ops.
///
/// Adding a new backend:
///   1. Add the enumerator to enum class Backend in include/core/policy.hpp
///   2. Create src/core/backends/<name>/ with impl.hpp and matrix.cpp
///   3. Add `case Backend::<name>:` to each switch below
///   4. Register the .cpp in cmake/sources.cmake

#include "core/matrix.hpp"
#include "core/parallel/cuda_ops.hpp"
#include <algorithm>

#include "backends/seq/impl.hpp"
#include "backends/blas/impl.hpp"
#include "backends/omp/impl.hpp"
#include "backends/gpu/impl.hpp"
#include "backends/simd/impl.hpp"

namespace num {

Matrix::Matrix(idx rows, idx cols)
    : rows_(rows)
    , cols_(cols)
    , data_(new real[rows * cols]()) {}

Matrix::Matrix(idx rows, idx cols, real val)
    : rows_(rows)
    , cols_(cols)
    , data_(new real[rows * cols]) {
    std::fill_n(data_.get(), size(), val);
}

Matrix::~Matrix() {
    if (d_data_)
        cuda::free(d_data_);
}

Matrix::Matrix(const Matrix& o)
    : rows_(o.rows_)
    , cols_(o.cols_)
    , data_(new real[o.size()]) {
    std::copy_n(o.data_.get(), size(), data_.get());
}

Matrix::Matrix(Matrix&& o) noexcept
    : rows_(o.rows_)
    , cols_(o.cols_)
    , data_(std::move(o.data_))
    , d_data_(o.d_data_) {
    o.rows_ = o.cols_ = 0;
    o.d_data_         = nullptr;
}

Matrix& Matrix::operator=(const Matrix& o) {
    if (this != &o) {
        rows_ = o.rows_;
        cols_ = o.cols_;
        data_.reset(new real[size()]);
        std::copy_n(o.data_.get(), size(), data_.get());
    }
    return *this;
}

Matrix& Matrix::operator=(Matrix&& o) noexcept {
    if (this != &o) {
        if (d_data_)
            cuda::free(d_data_);
        rows_   = o.rows_;
        cols_   = o.cols_;
        data_   = std::move(o.data_);
        d_data_ = o.d_data_;
        o.rows_ = o.cols_ = 0;
        o.d_data_         = nullptr;
    }
    return *this;
}

void Matrix::to_gpu() {
    if (!d_data_) {
        d_data_ = cuda::alloc(size());
        cuda::to_device(d_data_, data_.get(), size());
    }
}

void Matrix::to_cpu() {
    if (d_data_) {
        cuda::to_host(data_.get(), d_data_, size());
        cuda::free(d_data_);
        d_data_ = nullptr;
    }
}

void matmul(const Matrix& A, const Matrix& B, Matrix& C, Backend b) {
    switch (b) {
        case Backend::seq:
            backends::seq::matmul(A, B, C);
            break;
        case Backend::blocked:
            backends::seq::matmul_blocked(A, B, C, 64);
            break;
        case Backend::simd:
            backends::simd::matmul(A, B, C, 64);
            break;
        case Backend::lapack:
            [[fallthrough]]; // no LAPACK matmul; use BLAS
        case Backend::blas:
            backends::blas::matmul(A, B, C);
            break;
        case Backend::omp:
            backends::omp::matmul(A, B, C);
            break;
        case Backend::gpu:
            backends::gpu::matmul(A, B, C);
            break;
    }
}

void matvec(const Matrix& A, const Vector& x, Vector& y, Backend b) {
    switch (b) {
        case Backend::seq:
            backends::seq::matvec(A, x, y);
            break;
        case Backend::blocked:
            backends::seq::matvec(A, x, y);
            break;
        case Backend::simd:
            backends::simd::matvec(A, x, y);
            break;
        case Backend::lapack:
            [[fallthrough]]; // no LAPACK matvec; use BLAS
        case Backend::blas:
            backends::blas::matvec(A, x, y);
            break;
        case Backend::omp:
            backends::omp::matvec(A, x, y);
            break;
        case Backend::gpu:
            backends::gpu::matvec(A, x, y);
            break;
    }
}

void matadd(real          alpha,
            const Matrix& A,
            real          beta,
            const Matrix& B,
            Matrix&       C,
            Backend       b) {
    switch (b) {
        case Backend::seq:
        case Backend::blocked:
        case Backend::simd:
            backends::seq::matadd(alpha, A, beta, B, C);
            break;
        case Backend::lapack:
            [[fallthrough]]; // no LAPACK matadd; use BLAS
        case Backend::blas:
            backends::blas::matadd(alpha, A, beta, B, C);
            break;
        case Backend::omp:
            backends::omp::matadd(alpha, A, beta, B, C);
            break;
        case Backend::gpu:
            backends::seq::matadd(alpha, A, beta, B, C);
            break;
    }
}

void matmul_blocked(const Matrix& A,
                    const Matrix& B,
                    Matrix&       C,
                    idx           block_size) {
    backends::seq::matmul_blocked(A, B, C, block_size);
}

void matmul_register_blocked(const Matrix& A,
                             const Matrix& B,
                             Matrix&       C,
                             idx           block_size,
                             idx           reg_size) {
    backends::seq::matmul_register_blocked(A, B, C, block_size, reg_size);
}

void matmul_simd(const Matrix& A, const Matrix& B, Matrix& C, idx block_size) {
    backends::simd::matmul(A, B, C, block_size);
}

void matvec_simd(const Matrix& A, const Vector& x, Vector& y) {
    backends::simd::matvec(A, x, y);
}

} // namespace num
