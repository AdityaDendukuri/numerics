/// @file matrix.hpp
/// @brief Dense row-major matrix templated over scalar type T.
#pragma once

#include "core/types.hpp"

#include "container/parallel/cuda_ops.hpp"
#include "container/vector.hpp"
#include <algorithm>
#include <concepts>
#include <memory>
#include <type_traits>

namespace num {

/// @brief Dense row-major owning matrix.
template <std::floating_point T>
class BasicMatrix {
  public:
    using value_type = T;

    /// Construct an empty matrix.
    BasicMatrix() : rows_(0), cols_(0), data_(nullptr) {}

    /// Construct a zero-initialized rows-by-cols matrix.
    BasicMatrix(idx rows, idx cols)
        : rows_(rows), cols_(cols),
          data_((rows * cols > 0) ? new T[rows * cols]() : nullptr) {}

    /// Construct a rows-by-cols matrix filled with val.
    BasicMatrix(idx rows, idx cols, T val)
        : rows_(rows), cols_(cols),
          data_((rows * cols > 0) ? new T[rows * cols] : nullptr) {
        if (size() > 0) std::fill_n(data_.get(), size(), val);
    }

    ~BasicMatrix() {
#if defined(NUMERICS_HAS_CUDA)
        if constexpr (std::is_same_v<T, real>) {
            if (d_data_) {
                cuda::free(d_data_);
                d_data_ = nullptr;
            }
        }
#endif
    }

    BasicMatrix(const BasicMatrix &o)
        : rows_(o.rows_), cols_(o.cols_),
          data_((o.size() > 0 && o.data_) ? new T[o.size()] : nullptr) {
        if (size() > 0 && o.data_) {
            std::copy_n(o.data_.get(), size(), data_.get());
        }
    }

    BasicMatrix(BasicMatrix &&o) noexcept
        : rows_(o.rows_), cols_(o.cols_), data_(std::move(o.data_)), d_data_(o.d_data_) {
        o.rows_ = o.cols_ = 0;
        o.d_data_ = nullptr;
    }

    BasicMatrix &operator=(const BasicMatrix &o) {
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
                data_.reset(new T[size()]);
                std::copy_n(o.data_.get(), size(), data_.get());
            } else {
                data_.reset();
            }
        }
        return *this;
    }

    BasicMatrix &operator=(BasicMatrix &&o) noexcept {
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

    /// Return contiguous row-major host storage.
    T *data() { return data_.get(); }
    [[nodiscard]] const T *data() const { return data_.get(); }

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
    template <class X = Vector, class Y = Vector>
    void apply(const X &x, Y &y) const;

  private:
    idx rows_ = 0, cols_ = 0;
    std::unique_ptr<T[]> data_;
    T *d_data_ = nullptr;
};

/// @brief Double-precision dense matrix with full backend dispatch (CPU + GPU).
using Matrix = BasicMatrix<real>;

#if defined(NUMERICS_EXTERN_TEMPLATES)
extern template class BasicMatrix<double>;
#endif

} // namespace num
