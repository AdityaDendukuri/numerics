/// @file matrix.hpp
/// @brief Dense row-major matrix templated over scalar type T.
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

/// @brief Dense row-major owning matrix.
template <std::floating_point T>
class basic_mat {
  public:
    using value_type = T;

    /// Construct an empty matrix.
    basic_mat() : rows_(0), cols_(0), data_(nullptr) {}

    /// Construct a zero-initialized rows-by-cols matrix.
    basic_mat(idx rows, idx cols)
        : rows_(rows), cols_(cols), data_(make_aligned<T>(checked_size(rows, cols))) {}

    /// Construct a rows-by-cols matrix filled with val.
    basic_mat(idx rows, idx cols, T val)
        : rows_(rows), cols_(cols),
          data_(make_aligned_for_overwrite<T>(checked_size(rows, cols))) {
        if (size() > 0) std::fill_n(data_.get(), size(), val);
    }

    ~basic_mat() {
#if defined(NUMERICS_HAS_CUDA)
        if constexpr (std::is_same_v<T, real>) {
            if (d_data_) {
                cuda::free(d_data_);
                d_data_ = nullptr;
            }
        }
#endif
    }

    basic_mat(const basic_mat &o)
        : rows_(o.rows_), cols_(o.cols_),
          data_(make_aligned_for_overwrite<T>(o.data_ ? o.size() : 0)) {
        if (size() > 0 && o.data_) {
            std::copy_n(o.data_.get(), size(), data_.get());
        }
    }

    basic_mat(basic_mat &&o) noexcept
        : rows_(o.rows_), cols_(o.cols_), data_(std::move(o.data_)), d_data_(o.d_data_) {
        o.rows_ = o.cols_ = 0;
        o.d_data_ = nullptr;
    }

    basic_mat &operator=(const basic_mat &o) {
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

    basic_mat &operator=(basic_mat &&o) noexcept {
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

    /// Return the matrix dimensions and total element count.
    [[nodiscard]] constexpr idx rows() const noexcept { return rows_; }
    [[nodiscard]] constexpr idx cols() const noexcept { return cols_; }
    [[nodiscard]] constexpr idx size() const noexcept { return rows_ * cols_; }

    /// Return contiguous row-major host storage, aligned to `num::storage_alignment`.
    ///
    /// The row stride is `cols()`, so only row 0 is guaranteed to start on that
    /// boundary; the alignment claim is about the base pointer.
    T *data() noexcept { return assume_storage_aligned(data_.get()); }
    [[nodiscard]] const T *data() const noexcept { return assume_storage_aligned(data_.get()); }

    T &operator()(idx i, idx j) { return data_[(i * cols_) + j]; }
    T operator()(idx i, idx j) const { return data_[(i * cols_) + j]; }

    /// Copy host data to a lazily allocated device mirror.
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

    /// Copy the device mirror back to host and release device storage.
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

    /// Return device storage, or null when no mirror exists.
    T *gpu_data() { return d_data_; }
    [[nodiscard]] const T *gpu_data() const { return d_data_; }
    [[nodiscard]] bool on_gpu() const { return d_data_ != nullptr; }

    /// Operator protocol application: y <- A * x
    template <class X = vec, class Y = vec>
    void apply(const X &x, Y &y) const;

  private:
    /// Element count of a rows-by-cols matrix, rejecting a product that would wrap.
    ///
    /// The three SuiteSparse bindings already guard their int32 interfaces this
    /// way; without the check here a wrapped product allocates a short buffer
    /// that `operator()` then indexes past.
    static idx checked_size(idx rows, idx cols) {
        if (rows != 0 && cols > std::numeric_limits<idx>::max() / rows) {
            throw std::overflow_error("basic_mat: rows * cols exceeds the index range");
        }
        return rows * cols;
    }

    idx rows_ = 0, cols_ = 0;
    aligned_array<T> data_;
    T *d_data_ = nullptr;
};

/// @brief Double-precision dense matrix with full backend dispatch (CPU + GPU).
using mat = basic_mat<real>;

#if defined(NUMERICS_EXTERN_TEMPLATES)
extern template class basic_mat<double>;
#endif

} // namespace num
