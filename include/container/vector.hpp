/// @file container/vector.hpp
/// @brief Dense vector.
#pragma once

#include "container/concepts.hpp"
#include "container/util/aligned_storage.hpp"
#include "core/math/laws.hpp"
#include "core/math/operations.hpp"
#include "core/types.hpp"
#include "cuda/cuda_ops.hpp"
#include "kernel/kernel.hpp"
#include "omp/parallel_ops.hpp"
#include <algorithm>
#include <cmath>
#include <memory>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace num {

/// @brief Dense vector \f$x \in \mathbb{K}^n\f$.
template <typename T>
class vec {
  public:
    using value_type = T;

    /// @brief Construct an empty vector.
    vec() : n_(0), data_(nullptr) {}

    /// @brief Construct a vector of `n` zeros.
    explicit vec(idx n) : n_(n), data_(make_aligned<T>(n)) {}

    /// @brief Construct a vector of `n` entries equal to `val`.
    vec(idx n, T val) : n_(n), data_(make_aligned_for_overwrite<T>(n)) {
        if (n_ > 0) std::fill_n(data_.get(), n_, val);
    }

    /// @brief Construct a vector from an initializer list.
    vec(std::initializer_list<T> init)
        : n_(init.size()), data_(make_aligned_for_overwrite<T>(init.size())) {
        if (n_ > 0) std::copy(init.begin(), init.end(), data_.get());
    }

    /// @brief Construct a vector by copying a view.
    explicit vec(view<const T> values)
        : n_(values.size()), data_(make_aligned_for_overwrite<T>(values.size())) {
        if (n_ > 0) std::copy(values.begin(), values.end(), data_.get());
    }

    explicit vec(const std::vector<T> &values) : vec(std::span<const T>(values)) {}

    ~vec() {
#if defined(NUMERICS_HAS_CUDA)
        if constexpr (std::is_same_v<T, real>) {
            if (d_data_) {
                cuda::free(d_data_);
                d_data_ = nullptr;
            }
        }
#endif
    }

    vec(const vec &o)
        : n_(o.n_), data_(make_aligned_for_overwrite<T>(o.data_ ? o.n_ : 0)) {
        if (n_ > 0 && o.data_) {
            std::copy_n(o.data_.get(), n_, data_.get());
        }
    }

    vec(vec &&o) noexcept
        : n_(o.n_), data_(std::move(o.data_)), d_data_(o.d_data_) {
        o.n_ = 0;
        o.d_data_ = nullptr;
    }

    vec &operator=(const vec &o) {
        if (this != &o) {
#if defined(NUMERICS_HAS_CUDA)
            if constexpr (std::is_same_v<T, real>) {
                if (d_data_) {
                    cuda::free(d_data_);
                    d_data_ = nullptr;
                }
            }
#endif
            n_ = o.n_;
            if (n_ > 0 && o.data_) {
                data_ = make_aligned_for_overwrite<T>(n_);
                std::copy_n(o.data_.get(), n_, data_.get());
            } else {
                data_.reset();
            }
        }
        return *this;
    }

    vec &operator=(vec &&o) noexcept {
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

    /// @brief Return the length \f$n\f$.
    [[nodiscard]] constexpr idx size() const noexcept { return n_; }

    /// @brief Return the vector itself, for code written for fields.
    vec &as_vec() { return *this; }
    [[nodiscard]] const vec &as_vec() const { return *this; }

    /// @brief Return the storage; it is aligned to `storage_alignment`.
    T *data() noexcept { return assume_storage_aligned(data_.get()); }
    [[nodiscard]] const T *data() const noexcept { return assume_storage_aligned(data_.get()); }

    /// @brief Return a view of the whole vector.
    [[nodiscard]] view<T> span() noexcept { return {data(), n_}; }
    [[nodiscard]] view<const T> span() const noexcept { return {data(), n_}; }

    T &operator[](idx i) { return data_[i]; }
    T operator[](idx i) const { return data_[i]; }

    /// @brief Return the first or last entry; the vector must not be empty.
    T &front() { return data_[0]; }
    [[nodiscard]] const T &front() const { return data_[0]; }
    T &back() { return data_[n_ - 1]; }
    [[nodiscard]] const T &back() const { return data_[n_ - 1]; }

    T *begin() { return data_.get(); }
    T *end() { return data_.get() + n_; }
    [[nodiscard]] const T *begin() const { return data_.get(); }
    [[nodiscard]] const T *end() const { return data_.get() + n_; }

    /// @brief Copy the vector to the device, allocating a copy there on first use.
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

    /// @brief Copy the vector back from the device and free the device copy.
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

    /// @brief Return the device copy, or null when there is none.
    real *gpu_data() { return d_data_; }
    [[nodiscard]] const real *gpu_data() const { return d_data_; }
    [[nodiscard]] bool on_gpu() const { return d_data_ != nullptr; }

  private:
    idx n_;
    aligned_array<T> data_;
    real *d_data_ = nullptr; // GPU mirror (real-typed); always nullptr for T != real
};

template <typename T>
/// @brief Copy a vector into a view of the same length.
void copy_to(const vec<T> &source, view<T> destination) {
    if (source.size() != destination.size()) {
        throw std::invalid_argument("copy_to: vector sizes must match");
    }
    std::copy(source.begin(), source.end(), destination.begin());
}

template <typename T>
/// @brief Copy a vector into a `std::vector` of the same length.
void copy_to(const vec<T> &source, std::vector<T> &destination) {
    copy_to(source, view<T>(destination));
}

/// @brief View of a flat vector as \f$(x_i, y_i)\f$ pairs.
struct vec2_view {
    vec<real> &v;

    /// @brief Return the number of pairs.
    [[nodiscard]] idx size() const noexcept { return v.size() / 2; }

    real &x(idx i) noexcept { return v[2 * i]; }
    [[nodiscard]] real x(idx i) const noexcept { return v[2 * i]; }
    real &y(idx i) noexcept { return v[(2 * i) + 1]; }
    [[nodiscard]] real y(idx i) const noexcept { return v[(2 * i) + 1]; }
};

// Native mathematical operations lower to the dependency-free raw kernels, one
// block per thread. This is the path every generic solver takes: `num::cg` and
// friends call these CPOs and nothing else, so threading has to arrive here
// rather than through an explicit `num::omp::dot(x, y)` call, which no
// algorithm uses. Foreign vectors use the coordinate fallbacks or provide their
// own tag_invoke overloads without inheriting from a numerics type.
template <std::floating_point T>
inline void tag_invoke(math::scale_t, T alpha, vec<T> &vector) noexcept {
    T *data = vector.data();
    omp::parallel_apply(vector.size(), [data, alpha](idx offset, idx length) {
        kernel::scale(data + offset, alpha, length);
    });
}

template <std::floating_point T>
inline void tag_invoke(math::axpy_t, T alpha, const vec<T> &x, vec<T> &y) {
    if (x.size() != y.size()) {
        throw std::invalid_argument("math::axpy: vector dimensions must match");
    }
    const T *xd = x.data();
    T *yd = y.data();
    omp::parallel_apply(x.size(), [xd, yd, alpha](idx offset, idx length) {
        kernel::axpy(yd + offset, xd + offset, alpha, length);
    });
}

template <std::floating_point T>
inline void tag_invoke(math::linear_combination_t, T alpha, const vec<T> &x, T beta,
                       vec<T> &y) {
    if (x.size() != y.size()) {
        throw std::invalid_argument("math::linear_combination: vector dimensions must match");
    }
    // y <- alpha*x + beta*y
    const T *xd = x.data();
    T *yd = y.data();
    omp::parallel_apply(x.size(), [xd, yd, alpha, beta](idx offset, idx length) {
        kernel::axpby(yd + offset, xd + offset, alpha, beta, length);
    });
}

template <std::floating_point T>
[[nodiscard]] inline T tag_invoke(math::inner_t, const vec<T> &x, const vec<T> &y) {
    if (x.size() != y.size()) {
        throw std::invalid_argument("math::inner: vector dimensions must match");
    }
    const T *xd = x.data();
    const T *yd = y.data();
    return omp::parallel_reduce<T>(x.size(), [xd, yd](idx offset, idx length) {
        return kernel::dot(xd + offset, yd + offset, length);
    });
}

template <std::floating_point T>
[[nodiscard]] inline T tag_invoke(math::axpy_norm_sq_t, T alpha, const vec<T> &x,
                                  vec<T> &y) {
    if (x.size() != y.size()) {
        throw std::invalid_argument("math::axpy_norm_sq: vector dimensions must match");
    }
    // y <- y + alpha*x; return ||y||_2^2. Blocks are disjoint, so the update and
    // the reduction stay correct when they are split across threads.
    const T *xd = x.data();
    T *yd = y.data();
    return omp::parallel_reduce<T>(x.size(), [xd, yd, alpha](idx offset, idx length) {
        return kernel::axpy_norm_sq(yd + offset, xd + offset, alpha, length);
    });
}

template <std::floating_point T>
[[nodiscard]] inline T tag_invoke(math::norm_t, const vec<T> &vector) noexcept {
    const T *data = vector.data();
    const idx n = vector.size();
    const T squared = omp::parallel_reduce<T>(n, [data](idx offset, idx length) {
        return kernel::norm_sq(data + offset, length);
    });
    if (squared > T(0) && std::isfinite(squared)) {
        return std::sqrt(squared);
    }
    // Overflowed, underflowed, or not a number: fall back to the rescaled
    // single-threaded form, which handles all three.
    return kernel::norm(data, n);
}

static_assert(repr::contiguous<vec<real>>);
} // namespace num
