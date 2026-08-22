/// @file matrix.hpp
/// @brief Dense row-major matrix templated over scalar type T.
#pragma once

#include "core/parallel/cuda_ops.hpp"
#include "core/policy.hpp"
#include "core/vector.hpp"
#include <algorithm>
#include <concepts>
#include <memory>
#include <type_traits>

namespace num {

/// @brief Dense row-major owning matrix.
template <std::floating_point T>
class BasicMatrix {
  public:
    /// Construct an empty matrix.
    BasicMatrix() : rows_(0), cols_(0), data_(nullptr) {}

    /// Construct a zero-initialized rows-by-cols matrix.
    BasicMatrix(idx rows, idx cols) : rows_(rows), cols_(cols), data_(new T[rows * cols]()) {}

    /// Construct a rows-by-cols matrix filled with val.
    BasicMatrix(idx rows, idx cols, T val) : rows_(rows), cols_(cols), data_(new T[rows * cols]) {
        std::fill_n(data_.get(), size(), val);
    }

    ~BasicMatrix() {
        if constexpr (std::is_same_v<T, real>) {
            if (d_data_) {
                cuda::free(d_data_);
            }
        }
    }

    BasicMatrix(const BasicMatrix &o) : rows_(o.rows_), cols_(o.cols_), data_(new T[o.size()]) {
        std::copy_n(o.data_.get(), size(), data_.get());
    }

    BasicMatrix(BasicMatrix &&o) noexcept
        : rows_(o.rows_), cols_(o.cols_), data_(std::move(o.data_)), d_data_(o.d_data_) {
        o.rows_ = o.cols_ = 0;
        o.d_data_ = nullptr;
    }

    BasicMatrix &operator=(const BasicMatrix &o) {
        if (this != &o) {
            rows_ = o.rows_;
            cols_ = o.cols_;
            data_.reset(new T[size()]);
            std::copy_n(o.data_.get(), size(), data_.get());
        }
        return *this;
    }

    BasicMatrix &operator=(BasicMatrix &&o) noexcept {
        if (this != &o) {
            if constexpr (std::is_same_v<T, real>) {
                if (d_data_) {
                    cuda::free(d_data_);
                }
            }
            rows_ = o.rows_;
            cols_ = o.cols_;
            data_ = std::move(o.data_);
            d_data_ = o.d_data_;
            o.rows_ = o.cols_ = 0;
            o.d_data_ = nullptr;
        }
        return *this;
    }

    /// Return the matrix dimensions and total element count.
    [[nodiscard]] constexpr idx rows() const noexcept { return rows_; }
    [[nodiscard]] constexpr idx cols() const noexcept { return cols_; }
    [[nodiscard]] constexpr idx size() const noexcept { return rows_ * cols_; }

    /// Return contiguous row-major host storage.
    T *data() { return data_.get(); }
    [[nodiscard]] const T *data() const { return data_.get(); }

    T &operator()(idx i, idx j) { return data_[(i * cols_) + j]; }
    T operator()(idx i, idx j) const { return data_[(i * cols_) + j]; }

    /// Copy host data to a lazily allocated device mirror.
    void to_gpu() {
        if constexpr (std::is_same_v<T, real>) {
            if (!d_data_) {
                d_data_ = cuda::alloc(size());
                cuda::to_device(d_data_, data_.get(), size());
            }
        }
    }

    /// Copy the device mirror back to host and release device storage.
    void to_cpu() {
        if constexpr (std::is_same_v<T, real>) {
            if (d_data_) {
                cuda::to_host(data_.get(), d_data_, size());
                cuda::free(d_data_);
                d_data_ = nullptr;
            }
        }
    }

    /// Return device storage, or null when no mirror exists.
    T *gpu_data() { return d_data_; }
    [[nodiscard]] const T *gpu_data() const { return d_data_; }
    [[nodiscard]] bool on_gpu() const { return d_data_ != nullptr; }

  private:
    idx rows_ = 0, cols_ = 0;
    std::unique_ptr<T[]> data_;
    T *d_data_ = nullptr;
};

/// @brief Double-precision dense matrix with full backend dispatch (CPU + GPU).
using Matrix = BasicMatrix<real>;

extern template class BasicMatrix<double>;

/// @brief y = A * x
void matvec(const Matrix &A, const Vector &x, Vector &y, Backend b = default_backend);

/// @brief C = A * B
void matmul(const Matrix &A, const Matrix &B, Matrix &C, Backend b = default_backend);

/// @brief C = alpha*A + beta*B
void matadd(real alpha, const Matrix &A, real beta, const Matrix &B, Matrix &C,
            Backend b = default_backend);

/// @brief C = A * B  (cache-blocked)
///
/// Divides A, B, C into BLOCKxBLOCK tiles so the working set fits in L2 cache.
/// @param A Left factor.
/// @param B Right factor.
/// @param C Output matrix.
/// @param block_size  Tile edge length (default 64).
void matmul_blocked(const Matrix &A, const Matrix &B, Matrix &C, idx block_size = 64);

/// @brief C = A * B  (register-blocked)
///
/// Extends cache blocking with a REGxREG register tile inside each cache tile.
/// @param A Left factor.
/// @param B Right factor.
/// @param C Output matrix.
/// @param block_size  Cache tile edge (default 64).
/// @param reg_size    Register tile edge (default 4).
void matmul_register_blocked(const Matrix &A, const Matrix &B, Matrix &C, idx block_size = 64,
                             idx reg_size = 4);

/// @brief C = A * B  (SIMD-accelerated)
///
/// Dispatches at compile time: AVX-256 + FMA on x86, NEON on AArch64,
/// falls back to matmul_blocked if neither is available.
void matmul_simd(const Matrix &A, const Matrix &B, Matrix &C, idx block_size = 64);

/// @brief y = A * x  (SIMD-accelerated)
void matvec_simd(const Matrix &A, const Vector &x, Vector &y);

} // namespace num
