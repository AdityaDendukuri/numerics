/// @file vector.hpp
/// @brief Vector operations
#pragma once

#include "core/parallel/cuda_ops.hpp"
#include "core/policy.hpp"
#include "core/types.hpp"
#include <algorithm>
#include <memory>
#include <type_traits>

namespace num {

/// @brief Dense vector with optional GPU storage, templated over scalar type T.
///
/// Typical aliases:
///   - `num::Vector`  = BasicVector<real>    -- real-valued, full backend
///   dispatch including GPU
///   - `num::CVector` = BasicVector<cplx>    -- complex-valued, sequential only
///   (no GPU)
///
/// All member functions are defined inline so the template is usable across
/// translation units without explicit instantiation.
template <typename T> class BasicVector {
public:
  BasicVector() : n_(0), data_(nullptr) {}

  explicit BasicVector(idx n) : n_(n), data_(new T[n]()) {}

  BasicVector(idx n, T val) : n_(n), data_(new T[n]) {
    std::fill_n(data_.get(), n_, val);
  }

  BasicVector(std::initializer_list<T> init)
      : n_(init.size()), data_(new T[n_]) {
    std::copy(init.begin(), init.end(), data_.get());
  }

  ~BasicVector() {
    if constexpr (std::is_same_v<T, real>) {
      if (d_data_)
        cuda::free(d_data_);
    }
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
      if constexpr (std::is_same_v<T, real>) {
        if (d_data_)
          cuda::free(d_data_);
      }
      n_ = o.n_;
      data_ = std::move(o.data_);
      d_data_ = o.d_data_;
      o.n_ = 0;
      o.d_data_ = nullptr;
    }
    return *this;
  }

  constexpr idx size() const noexcept { return n_; }

  /// Satisfy the VecField concept: a Vector is its own underlying vector.
  BasicVector &vec() { return *this; }
  const BasicVector &vec() const { return *this; }

  T *data() { return data_.get(); }
  const T *data() const { return data_.get(); }

  T &operator[](idx i) { return data_[i]; }
  T operator[](idx i) const { return data_[i]; }

  T *begin() { return data_.get(); }
  T *end() { return data_.get() + n_; }
  const T *begin() const { return data_.get(); }
  const T *end() const { return data_.get() + n_; }

  // GPU lifecycle (no-op for T != real)

  void to_gpu() {
    if constexpr (std::is_same_v<T, real>) {
      if (!d_data_) {
        d_data_ = cuda::alloc(n_);
        cuda::to_device(d_data_, data_.get(), n_);
      }
    }
  }

  void to_cpu() {
    if constexpr (std::is_same_v<T, real>) {
      if (d_data_) {
        cuda::to_host(data_.get(), d_data_, n_);
        cuda::free(d_data_);
        d_data_ = nullptr;
      }
    }
  }

  real *gpu_data() { return d_data_; }
  const real *gpu_data() const { return d_data_; }
  bool on_gpu() const { return d_data_ != nullptr; }

private:
  idx n_;
  std::unique_ptr<T[]> data_;
  real *d_data_ =
      nullptr; // GPU mirror (real-typed); always nullptr for T != real
};

/// @brief Real-valued dense vector with full backend dispatch (CPU + GPU)
using Vector = BasicVector<real>;

/// @brief Complex-valued dense vector (sequential; no GPU)
using CVector = BasicVector<cplx>;

// Real-vector free functions (full backend dispatch)

/// @brief v *= alpha
void scale(Vector &v, real alpha, Backend b = default_backend);

/// @brief z = x + y
void add(const Vector &x, const Vector &y, Vector &z,
         Backend b = default_backend);

/// @brief y += alpha * x
void axpy(real alpha, const Vector &x, Vector &y, Backend b = default_backend);

/// @brief dot product
real dot(const Vector &x, const Vector &y, Backend b = default_backend);

/// @brief Euclidean norm
real norm(const Vector &x, Backend b = default_backend);

// Strided views

/// @brief Non-owning view of a flat Vector as an array of 2D points.
///
/// Many physics simulations pack 2D positions or velocities into a flat
/// Vector with stride 2: element i has x = v[2i], y = v[2i+1].
/// Vec2View provides named accessors that eliminate the raw index arithmetic
/// without copying or reshaping the data.
///
/// Example:
/// @code
///   num::Vector q(2 * N);          // flat storage: [x0,y0, x1,y1, ...]
///   num::Vec2View pos{q};
///   pos.x(i) = 1.0;               // writes q[2*i]
///   pos.y(i) = 2.0;               // writes q[2*i+1]
///   idx n = pos.size();            // = N
/// @endcode
struct Vec2View {
  Vector &v;

  /// Number of 2D points (= v.size() / 2).
  idx size() const noexcept { return v.size() / 2; }

  real &x(idx i) noexcept { return v[2 * i]; }
  real x(idx i) const noexcept { return v[2 * i]; }
  real &y(idx i) noexcept { return v[2 * i + 1]; }
  real y(idx i) const noexcept { return v[2 * i + 1]; }
};

/// @brief Read-only variant of Vec2View for const Vectors.
struct Vec2ConstView {
  const Vector &v;

  idx size() const noexcept { return v.size() / 2; }
  real x(idx i) const noexcept { return v[2 * i]; }
  real y(idx i) const noexcept { return v[2 * i + 1]; }
};

// Complex-vector free functions (sequential)

/// @brief v *= alpha
void scale(CVector &v, cplx alpha);

/// @brief y += alpha * x
void axpy(cplx alpha, const CVector &x, CVector &y);

/// @brief Conjugate inner product <x, y> = Sigma conj(x_i) * y_i
cplx dot(const CVector &x, const CVector &y);

/// @brief Euclidean norm  sqrt(Sigma |v_i|^2)
real norm(const CVector &x);

} // namespace num
