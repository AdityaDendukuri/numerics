/// @file vector.hpp
/// @brief Dense vector storage and operations.
#pragma once

#include "container/parallel/cuda_ops.hpp"
#include "core/math/models.hpp"
#include "core/math/operations.hpp"
#include "core/types.hpp"
#include "kernel/raw.hpp"
#include <algorithm>
#include <memory>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace num {

/// @brief Dense owning vector.
template <typename T>
class BasicVector {
  public:
    using value_type = T;

    /// Construct an empty vector.
    BasicVector() : n_(0), data_(nullptr) {}

    /// Construct n value-initialized elements.
    explicit BasicVector(idx n) : n_(n), data_(new T[n]()) {}

    /// Construct n elements initialized to val.
    BasicVector(idx n, T val) : n_(n), data_(new T[n]) { std::fill_n(data_.get(), n_, val); }

    /// Copy values from an initializer list.
    BasicVector(std::initializer_list<T> init) : n_(init.size()), data_(new T[n_]) {
        std::copy(init.begin(), init.end(), data_.get());
    }

    /// Copy values from a non-owning span.
    explicit BasicVector(std::span<const T> values) : n_(values.size()), data_(new T[n_]) {
        std::copy(values.begin(), values.end(), data_.get());
    }

    explicit BasicVector(const std::vector<T> &values) : BasicVector(std::span<const T>(values)) {}

    ~BasicVector() {
#if defined(NUMERICS_HAS_CUDA)
        if constexpr (std::is_same_v<T, real>) {
            if (d_data_) {
                cuda::free(d_data_);
            }
        }
#endif
    }

    BasicVector(const BasicVector &o) : n_(o.n_), data_(new T[n_]) {
        std::copy_n(o.data_.get(), n_, data_.get());
    }

    BasicVector(BasicVector &&o) noexcept
        : n_(o.n_), data_(std::move(o.data_)), d_data_(o.d_data_) {
        o.n_ = 0;
        o.d_data_ = nullptr;
    }

    BasicVector &operator=(const BasicVector &o) {
        if (this != &o) {
            n_ = o.n_;
            data_.reset(new T[n_]);
            std::copy_n(o.data_.get(), n_, data_.get());
        }
        return *this;
    }

    BasicVector &operator=(BasicVector &&o) noexcept {
        if (this != &o) {
#if defined(NUMERICS_HAS_CUDA)
            if constexpr (std::is_same_v<T, real>) {
                if (d_data_) {
                    cuda::free(d_data_);
                }
            }
#endif
            n_ = o.n_;
            data_ = std::move(o.data_);
            d_data_ = o.d_data_;
            o.n_ = 0;
            o.d_data_ = nullptr;
        }
        return *this;
    }

    /// Return the element count.
    [[nodiscard]] constexpr idx size() const noexcept { return n_; }

    /// Expose the vector itself for field-compatible generic code.
    BasicVector &vec() { return *this; }
    [[nodiscard]] const BasicVector &vec() const { return *this; }

    /// Return contiguous host storage.
    T *data() { return data_.get(); }
    [[nodiscard]] const T *data() const { return data_.get(); }

    T &operator[](idx i) { return data_[i]; }
    T operator[](idx i) const { return data_[i]; }

    T *begin() { return data_.get(); }
    T *end() { return data_.get() + n_; }
    [[nodiscard]] const T *begin() const { return data_.get(); }
    [[nodiscard]] const T *end() const { return data_.get() + n_; }

    /// Copy host data to a lazily allocated device mirror.
    void to_gpu() {
#if defined(NUMERICS_HAS_CUDA)
        if constexpr (std::is_same_v<T, real>) {
            if (!d_data_) {
                d_data_ = cuda::alloc(n_);
                cuda::to_device(d_data_, data_.get(), n_);
            }
        }
#endif
    }

    /// Copy the device mirror back to host and release device storage.
    void to_cpu() {
#if defined(NUMERICS_HAS_CUDA)
        if constexpr (std::is_same_v<T, real>) {
            if (d_data_) {
                cuda::to_host(data_.get(), d_data_, n_);
                cuda::free(d_data_);
                d_data_ = nullptr;
            }
        }
#endif
    }

    /// Return device storage, or null when no mirror exists.
    real *gpu_data() { return d_data_; }
    [[nodiscard]] const real *gpu_data() const { return d_data_; }
    [[nodiscard]] bool on_gpu() const { return d_data_ != nullptr; }

  private:
    idx n_;
    std::unique_ptr<T[]> data_;
    real *d_data_ = nullptr; // GPU mirror (real-typed); always nullptr for T != real
};

template <typename T>
/// Copy an owning vector into an equally sized span.
void copy_to(const BasicVector<T> &source, std::span<T> destination) {
    if (source.size() != destination.size()) {
        throw std::invalid_argument("copy_to: vector sizes must match");
    }
    std::copy(source.begin(), source.end(), destination.begin());
}

template <typename T>
/// Copy an owning vector into an equally sized std::vector.
void copy_to(const BasicVector<T> &source, std::vector<T> &destination) {
    copy_to(source, std::span<T>(destination));
}

/// @brief Real-valued dense vector with full backend dispatch (CPU + GPU)
using Vector = BasicVector<real>;

#if defined(NUMERICS_EXTERN_TEMPLATES)
// Defined by the numerics build so its own translation units share one
// instantiation. Left undefined when these headers are copied out on their own,
// where implicit instantiation is what makes the header usable without linking.
extern template class BasicVector<double>;
#endif

/// @brief Complex-valued dense vector (sequential; no GPU)
using CVector = BasicVector<cplx>;

/// @brief Non-owning view of a flat vector as \f$(x_i,y_i)\f$ pairs.
struct Vec2View {
    Vector &v;

    /// Return the number of coordinate pairs.
    [[nodiscard]] idx size() const noexcept { return v.size() / 2; }

    real &x(idx i) noexcept { return v[2 * i]; }
    [[nodiscard]] real x(idx i) const noexcept { return v[2 * i]; }
    real &y(idx i) noexcept { return v[(2 * i) + 1]; }
    [[nodiscard]] real y(idx i) const noexcept { return v[(2 * i) + 1]; }
};

// Native mathematical operations lower to the dependency-free raw kernels.
// Foreign vectors use the coordinate fallbacks or provide their own tag_invoke
// overloads without inheriting from a numerics type.
template <std::floating_point T>
inline void tag_invoke(math::scale_t, T alpha, BasicVector<T> &vector) noexcept {
    kernel::raw::scale(vector.data(), alpha, vector.size());
}

template <std::floating_point T>
inline void tag_invoke(math::axpy_t, T alpha, const BasicVector<T> &x, BasicVector<T> &y) {
    if (x.size() != y.size()) {
        throw std::invalid_argument("math::axpy: vector dimensions must match");
    }
    kernel::raw::axpy(y.data(), x.data(), alpha, x.size());
}

template <std::floating_point T>
inline void tag_invoke(math::linear_combination_t, T alpha, const BasicVector<T> &x, T beta,
                       BasicVector<T> &y) {
    if (x.size() != y.size()) {
        throw std::invalid_argument("math::linear_combination: vector dimensions must match");
    }
    // y <- alpha*x + beta*y
    kernel::raw::axpby(y.data(), x.data(), alpha, beta, x.size());
}

template <std::floating_point T>
[[nodiscard]] inline T tag_invoke(math::inner_t, const BasicVector<T> &x, const BasicVector<T> &y) {
    if (x.size() != y.size()) {
        throw std::invalid_argument("math::inner: vector dimensions must match");
    }
    return kernel::raw::dot(x.data(), y.data(), x.size());
}

template <std::floating_point T>
[[nodiscard]] inline T tag_invoke(math::axpy_norm_sq_t, T alpha, const BasicVector<T> &x,
                                  BasicVector<T> &y) {
    if (x.size() != y.size()) {
        throw std::invalid_argument("math::axpy_norm_sq: vector dimensions must match");
    }
    // y <- y + alpha*x; return ||y||_2^2
    return kernel::raw::axpy_norm_sq(y.data(), x.data(), alpha, x.size());
}

template <std::floating_point T>
[[nodiscard]] inline T tag_invoke(math::norm_t, const BasicVector<T> &vector) noexcept {
    return kernel::raw::norm(vector.data(), vector.size());
}

namespace math {

template <class T>
requires Models<T, law::field> struct model_of<BasicVector<T>> {
    using laws = type_list<law::inner_product_space>;
};

} // namespace math

} // namespace num
