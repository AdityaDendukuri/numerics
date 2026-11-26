/// @file container/matrix.hpp
/// @brief Dense row-major matrix.
#pragma once

#include "core/types.hpp"

#include "cuda/cuda_ops.hpp"
#include "container/util/aligned_storage.hpp"
#include "container/vector.hpp"
#include <algorithm>
#include <concepts>
#include <limits>
#include <memory>
#include <stdexcept>
#include <type_traits>

namespace num {

/// @brief Dense row-major matrix \f$A \in \mathbb{K}^{m \times n}\f$.
template <std::floating_point T>
class mat {
  public:
    using value_type = T;

    /// @brief Construct an empty matrix.
    mat() : rows_(0), cols_(0), data_(nullptr) {}

    /// @brief Construct a `rows x cols` matrix of zeros.
    mat(idx rows, idx cols)
        : rows_(rows), cols_(cols), data_(make_aligned<T>(checked_size(rows, cols))) {}

    /// @brief Construct a `rows x cols` matrix with every entry equal to `val`.
    mat(idx rows, idx cols, T val)
        : rows_(rows), cols_(cols),
          data_(make_aligned_for_overwrite<T>(checked_size(rows, cols))) {
        if (size() > 0) std::fill_n(data_.get(), size(), val);
    }

    ~mat() {
#if defined(NUMERICS_HAS_CUDA)
        if constexpr (std::is_same_v<T, real>) {
            if (d_data_) {
                cuda::free(d_data_);
                d_data_ = nullptr;
            }
        }
#endif
    }

    mat(const mat &o)
        : rows_(o.rows_), cols_(o.cols_),
          data_(make_aligned_for_overwrite<T>(o.data_ ? o.size() : 0)) {
        if (size() > 0 && o.data_) {
            std::copy_n(o.data_.get(), size(), data_.get());
        }
    }

    mat(mat &&o) noexcept
        : rows_(o.rows_), cols_(o.cols_), data_(std::move(o.data_)), d_data_(o.d_data_) {
        o.rows_ = o.cols_ = 0;
        o.d_data_ = nullptr;
    }

    mat &operator=(const mat &o) {
        if (this != &o) {
#if defined(NUMERICS_HAS_CUDA)
            if constexpr (std::is_same_v<T, real>) {
                if (d_data_) {
                    cuda::free(d_data_);
                    d_data_ = nullptr;
                }
            }
#endif
            rows_ = o.rows_;
            cols_ = o.cols_;
            if (size() > 0 && o.data_) {
                data_ = make_aligned_for_overwrite<T>(size());
                std::copy_n(o.data_.get(), size(), data_.get());
            } else {
                data_.reset();
            }
        }
        return *this;
    }

    mat &operator=(mat &&o) noexcept {
        if (this != &o) {
#if defined(NUMERICS_HAS_CUDA)
            if constexpr (std::is_same_v<T, real>) {
                if (d_data_) {
                    cuda::free(d_data_);
                }
            }
#endif
            rows_ = o.rows_;
            cols_ = o.cols_;
            data_ = std::move(o.data_);
            d_data_ = o.d_data_;
            o.rows_ = o.cols_ = 0;
            o.d_data_ = nullptr;
        }
        return *this;
    }

    /// @brief Return the number of rows, columns and entries.
    [[nodiscard]] constexpr idx rows() const noexcept { return rows_; }
    [[nodiscard]] constexpr idx cols() const noexcept { return cols_; }
    [[nodiscard]] constexpr idx size() const noexcept { return rows_ * cols_; }

    /// @brief Return the row-major storage; the base pointer is aligned to `storage_alignment`.
    /// @note Rows after the first are not guaranteed to start on an aligned boundary.
    T *data() noexcept { return assume_storage_aligned(data_.get()); }
    [[nodiscard]] const T *data() const noexcept { return assume_storage_aligned(data_.get()); }

    T &operator()(idx i, idx j) { return data_[(i * cols_) + j]; }
    T operator()(idx i, idx j) const { return data_[(i * cols_) + j]; }

    /// @brief Copy the matrix to the device, allocating a copy there on first use.
    void to_gpu() {
#if defined(NUMERICS_HAS_CUDA)
        if constexpr (std::is_same_v<T, real>) {
            if (!d_data_) {
                d_data_ = cuda::alloc(size());
                cuda::to_device(d_data_, data_.get(), size());
            }
        }
#endif
    }

    /// @brief Copy the matrix back from the device and free the device copy.
    void to_cpu() {
#if defined(NUMERICS_HAS_CUDA)
        if constexpr (std::is_same_v<T, real>) {
            if (d_data_) {
                cuda::to_host(data_.get(), d_data_, size());
                cuda::free(d_data_);
                d_data_ = nullptr;
            }
        }
#endif
    }

    /// @brief Return the device copy, or null when there is none.
    T *gpu_data() { return d_data_; }
    [[nodiscard]] const T *gpu_data() const { return d_data_; }
    [[nodiscard]] bool on_gpu() const { return d_data_ != nullptr; }

    /// @brief Apply the matrix to a vector: \f$y \leftarrow A x\f$.
    template <class X = vec<real>, class Y = vec<real>>
    void apply(const X &x, Y &y) const;

  private:
    /// @brief Return the number of entries \f$m n\f$, rejecting overflow.
    static idx checked_size(idx rows, idx cols) {
        if (rows != 0 && cols > std::numeric_limits<idx>::max() / rows) {
            throw std::overflow_error("mat: rows * cols exceeds the index range");
        }
        return rows * cols;
    }

    idx rows_ = 0, cols_ = 0;
    aligned_array<T> data_;
    T *d_data_ = nullptr;
};

} // namespace num
