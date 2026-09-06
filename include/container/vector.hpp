/// @file vector.hpp
/// @brief Dense vector storage and operations.
#pragma once

#include "cuda/cuda_ops.hpp"
#include "container/util/aligned_storage.hpp"
#include "omp/parallel_ops.hpp"
#include "core/math/models.hpp"
#include "core/math/operations.hpp"
#include "core/types.hpp"
#include "kernel/kernel.hpp"
#include <algorithm>
#include <cmath>
#include <memory>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace num {

/// @brief Dense owning vector.
template <typename T>
class basic_vec {
  public:
    using value_type = T;

    /// Construct an empty vector.
    basic_vec() : n_(0), data_(nullptr) {}

    /// Construct n value-initialized elements.
    explicit basic_vec(idx n) : n_(n), data_(make_aligned<T>(n)) {}

    /// Construct n elements initialized to val.
    basic_vec(idx n, T val) : n_(n), data_(make_aligned_for_overwrite<T>(n)) {
        if (n_ > 0) std::fill_n(data_.get(), n_, val);
    }

    /// Copy values from an initializer list.
    basic_vec(std::initializer_list<T> init)
        : n_(init.size()), data_(make_aligned_for_overwrite<T>(init.size())) {
        if (n_ > 0) std::copy(init.begin(), init.end(), data_.get());
    }

    /// Copy values from a non-owning span.
    explicit basic_vec(view<const T> values)
        : n_(values.size()), data_(make_aligned_for_overwrite<T>(values.size())) {
        if (n_ > 0) std::copy(values.begin(), values.end(), data_.get());
    }

    explicit basic_vec(const std::vector<T> &values) : basic_vec(std::span<const T>(values)) {}

    ~basic_vec() {
#if defined(NUMERICS_HAS_CUDA)
        if constexpr (std::is_same_v<T, real>) {
            if (d_data_) {
                cuda::free(d_data_);
                d_data_ = nullptr;
            }
        }
#endif
    }

    basic_vec(const basic_vec &o)
        : n_(o.n_), data_(make_aligned_for_overwrite<T>(o.data_ ? o.n_ : 0)) {
        if (n_ > 0 && o.data_) {
            std::copy_n(o.data_.get(), n_, data_.get());
        }
    }

    basic_vec(basic_vec &&o) noexcept
        : n_(o.n_), data_(std::move(o.data_)), d_data_(o.d_data_) {
        o.n_ = 0;
        o.d_data_ = nullptr;
    }

    basic_vec &operator=(const basic_vec &o) {
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

    basic_vec &operator=(basic_vec &&o) noexcept {
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
    basic_vec &as_vec() { return *this; }
    [[nodiscard]] const basic_vec &as_vec() const { return *this; }

    /// Return contiguous host storage, aligned to `num::storage_alignment`.
    ///
    /// The `assume_storage_aligned` wrapper is what carries the guarantee into
    /// the caller's codegen; the aligned allocation on its own is invisible to
    /// the optimizer across a function boundary.
    T *data() noexcept { return assume_storage_aligned(data_.get()); }
    [[nodiscard]] const T *data() const noexcept { return assume_storage_aligned(data_.get()); }

    /// @brief A view over the whole vector, for interfaces that take `std::span`.
    ///
    /// The storage is contiguous, so this is a pointer and a length. It exists because
    /// several routines (`sample_categorical`, `categorical_sampler`, the diagonal
    /// helpers) accept a span, and spelling that out at every call site is noise.
    [[nodiscard]] view<T> span() noexcept { return {data(), n_}; }
    [[nodiscard]] view<const T> span() const noexcept { return {data(), n_}; }

    T &operator[](idx i) { return data_[i]; }
    T operator[](idx i) const { return data_[i]; }

    /// @brief First and last elements. Undefined when the vector is empty, as for any
    /// sequence container.
    T &front() { return data_[0]; }
    [[nodiscard]] const T &front() const { return data_[0]; }
    T &back() { return data_[n_ - 1]; }
    [[nodiscard]] const T &back() const { return data_[n_ - 1]; }

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
    aligned_array<T> data_;
    real *d_data_ = nullptr; // GPU mirror (real-typed); always nullptr for T != real
};

template <typename T>
/// Copy an owning vector into an equally sized span.
void copy_to(const basic_vec<T> &source, view<T> destination) {
    if (source.size() != destination.size()) {
        throw std::invalid_argument("copy_to: vector sizes must match");
    }
    std::copy(source.begin(), source.end(), destination.begin());
}

template <typename T>
/// Copy an owning vector into an equally sized std::vector.
void copy_to(const basic_vec<T> &source, std::vector<T> &destination) {
    copy_to(source, view<T>(destination));
}

/// @brief Real-valued dense vector with full backend dispatch (CPU + GPU)
using vec = basic_vec<real>;

#if defined(NUMERICS_EXTERN_TEMPLATES)
// Defined by the numerics build so its own translation units share one
// instantiation. Left undefined when these headers are copied out on their own,
// where implicit instantiation is what makes the header usable without linking.
extern template class basic_vec<double>;
#endif

/// @brief Complex-valued dense vector (sequential; no GPU)
using cvec = basic_vec<cplx>;

/// @brief Non-owning view of a flat vector as \f$(x_i,y_i)\f$ pairs.
struct vec2_view {
    vec &v;

    /// Return the number of coordinate pairs.
    [[nodiscard]] idx size() const noexcept { return v.size() / 2; }

    real &x(idx i) noexcept { return v[2 * i]; }
    [[nodiscard]] real x(idx i) const noexcept { return v[2 * i]; }
    real &y(idx i) noexcept { return v[(2 * i) + 1]; }
    [[nodiscard]] real y(idx i) const noexcept { return v[(2 * i) + 1]; }
};

// Native mathematical operations lower to the dependency-free raw kernels, one
// block per thread. This is the path every generic solver takes: `math_cg` and
// friends call these CPOs and nothing else, so threading has to arrive here
// rather than through an explicit `num::omp::dot(x, y)` call, which no
// algorithm uses. Foreign vectors use the coordinate fallbacks or provide their
// own tag_invoke overloads without inheriting from a numerics type.
template <std::floating_point T>
inline void tag_invoke(math::scale_t, T alpha, basic_vec<T> &vector) noexcept {
    T *data = vector.data();
    omp::parallel_apply(vector.size(), [data, alpha](idx offset, idx length) {
        kernel::scale(data + offset, alpha, length);
    });
}

template <std::floating_point T>
inline void tag_invoke(math::axpy_t, T alpha, const basic_vec<T> &x, basic_vec<T> &y) {
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
inline void tag_invoke(math::linear_combination_t, T alpha, const basic_vec<T> &x, T beta,
                       basic_vec<T> &y) {
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
[[nodiscard]] inline T tag_invoke(math::inner_t, const basic_vec<T> &x, const basic_vec<T> &y) {
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
[[nodiscard]] inline T tag_invoke(math::axpy_norm_sq_t, T alpha, const basic_vec<T> &x,
                                  basic_vec<T> &y) {
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
[[nodiscard]] inline T tag_invoke(math::norm_t, const basic_vec<T> &vector) noexcept {
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

namespace math {

template <class T>
requires claims<T, law::field> struct claims_of<basic_vec<T>> {
    // `hilbert_space`, not merely `inner_product_space`: `norm` here is
    // sqrt(inner(x, x)), the norm the inner product induces, which is exactly what
    // separates the two laws. Every Krylov method relies on that identity, so claiming
    // only the weaker law would leave `hilbert_space<vec>` false.
    using type = type_list<law::hilbert_space>;
};

} // namespace math

} // namespace num
