/// @file vector.hpp
/// @brief Dense vector storage and operations.
#pragma once

#include "core/parallel/cuda_ops.hpp"
#include "core/policy.hpp"
#include "core/types.hpp"
#include <algorithm>
#include <memory>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace num {

/// @brief Dense owning vector.
template<typename T>
class BasicVector {
public:
  BasicVector()
      : n_(0),
        data_(nullptr) {}

  explicit BasicVector(idx n)
      : n_(n),
        data_(new T[n]()) {}

  BasicVector(idx n, T val)
      : n_(n),
        data_(new T[n]) {
    std::fill_n(data_.get(), n_, val);
  }

  BasicVector(std::initializer_list<T> init)
      : n_(init.size()),
        data_(new T[n_]) {
    std::copy(init.begin(), init.end(), data_.get());
  }

  explicit BasicVector(std::span<const T> values)
      : n_(values.size()),
        data_(new T[n_]) {
    std::copy(values.begin(), values.end(), data_.get());
  }

  explicit BasicVector(const std::vector<T>& values)
      : BasicVector(std::span<const T>(values)) {}

  ~BasicVector() {
    if constexpr (std::is_same_v<T, real>) {
      if (d_data_) {
        cuda::free(d_data_);
      }
    }
  }

  BasicVector(const BasicVector& o)
      : n_(o.n_),
        data_(new T[n_]) {
    std::copy_n(o.data_.get(), n_, data_.get());
  }

  BasicVector(BasicVector&& o) noexcept
      : n_(o.n_),
        data_(std::move(o.data_)),
        d_data_(o.d_data_) {
    o.n_ = 0;
    o.d_data_ = nullptr;
  }

  BasicVector& operator=(const BasicVector& o) {
    if (this != &o) {
      n_ = o.n_;
      data_.reset(new T[n_]);
      std::copy_n(o.data_.get(), n_, data_.get());
    }
    return *this;
  }

  BasicVector& operator=(BasicVector&& o) noexcept {
    if (this != &o) {
      if constexpr (std::is_same_v<T, real>) {
        if (d_data_) {
          cuda::free(d_data_);
        }
      }
      n_ = o.n_;
      data_ = std::move(o.data_);
      d_data_ = o.d_data_;
      o.n_ = 0;
      o.d_data_ = nullptr;
    }
    return *this;
  }

  [[nodiscard]] constexpr idx size() const noexcept { return n_; }

  BasicVector& vec() { return *this; }
  [[nodiscard]] const BasicVector& vec() const { return *this; }

  T* data() { return data_.get(); }
  [[nodiscard]] const T* data() const { return data_.get(); }

  T& operator[](idx i) { return data_[i]; }
  T operator[](idx i) const { return data_[i]; }

  T* begin() { return data_.get(); }
  T* end() { return data_.get() + n_; }
  [[nodiscard]] const T* begin() const { return data_.get(); }
  [[nodiscard]] const T* end() const { return data_.get() + n_; }

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

  real* gpu_data() { return d_data_; }
  [[nodiscard]] const real* gpu_data() const { return d_data_; }
  [[nodiscard]] bool on_gpu() const { return d_data_ != nullptr; }

private:
  idx n_;
  std::unique_ptr<T[]> data_;
  real* d_data_ = nullptr; // GPU mirror (real-typed); always nullptr for T != real
};

template<typename T>
void copy_to(const BasicVector<T>& source, std::span<T> destination) {
  if (source.size() != destination.size()) {
    throw std::invalid_argument("copy_to: vector sizes must match");
  }
  std::copy(source.begin(), source.end(), destination.begin());
}

template<typename T>
void copy_to(const BasicVector<T>& source, std::vector<T>& destination) {
  copy_to(source, std::span<T>(destination));
}

/// @brief Real-valued dense vector with full backend dispatch (CPU + GPU)
using Vector = BasicVector<real>;

extern template class BasicVector<double>;

/// @brief Complex-valued dense vector (sequential; no GPU)
using CVector = BasicVector<cplx>;

/// @brief Compute \f$v \leftarrow \alpha v\f$.
void scale(Vector& v, real alpha, Backend b = default_backend);

/// @brief Compute \f$z=x+y\f$.
void add(const Vector& x, const Vector& y, Vector& z, Backend b = default_backend);

/// @brief Compute \f$y \leftarrow y+\alpha x\f$.
void axpy(real alpha, const Vector& x, Vector& y, Backend b = default_backend);

/// @brief Compute \f$x^T y\f$.
real dot(const Vector& x, const Vector& y, Backend b = default_backend);

/// @brief Compute a sequential dot product over non-owning vectors.
inline real dot(std::span<const real> x, std::span<const real> y) {
  if (x.size() != y.size()) {
    throw std::invalid_argument("dot: vector sizes must match");
  }
  real result = 0.0;
  for (idx index = 0; index < x.size(); ++index) {
    result += x[index] * y[index];
  }
  return result;
}

/// @brief Compute \f$\|x\|_2\f$.
real norm(const Vector& x, Backend b = default_backend);

/// @brief Non-owning view of a flat vector as \f$(x_i,y_i)\f$ pairs.
struct Vec2View {
  Vector& v;

  [[nodiscard]] idx size() const noexcept { return v.size() / 2; }

  real& x(idx i) noexcept { return v[2 * i]; }
  [[nodiscard]] real x(idx i) const noexcept { return v[2 * i]; }
  real& y(idx i) noexcept { return v[(2 * i) + 1]; }
  [[nodiscard]] real y(idx i) const noexcept { return v[(2 * i) + 1]; }
};

struct Vec2ConstView {
  const Vector& v;

  [[nodiscard]] idx size() const noexcept { return v.size() / 2; }
  [[nodiscard]] real x(idx i) const noexcept { return v[2 * i]; }
  [[nodiscard]] real y(idx i) const noexcept { return v[(2 * i) + 1]; }
};

/// @brief v *= alpha
void scale(CVector& v, cplx alpha);

/// @brief y += alpha * x
void axpy(cplx alpha, const CVector& x, CVector& y);

/// @brief Conjugate inner product <x, y> = Sigma conj(x_i) * y_i
cplx dot(const CVector& x, const CVector& y);

/// @brief Euclidean norm  sqrt(Sigma |v_i|^2)
real norm(const CVector& x);

} // namespace num
