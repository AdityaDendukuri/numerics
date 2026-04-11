/// @file matrix.hpp
/// @brief Matrix operations
#pragma once

#include "core/policy.hpp"
#include "core/small_matrix.hpp"
#include "core/vector.hpp"

namespace num {

/// @brief Dense row-major matrix with optional GPU storage
class Matrix {
public:
  Matrix() : rows_(0), cols_(0), data_(nullptr) {}
  Matrix(idx rows, idx cols);
  Matrix(idx rows, idx cols, real val);
  ~Matrix();

  Matrix(const Matrix &);
  Matrix(Matrix &&) noexcept;
  Matrix &operator=(const Matrix &);
  Matrix &operator=(Matrix &&) noexcept;

  constexpr idx rows() const noexcept { return rows_; }
  constexpr idx cols() const noexcept { return cols_; }
  constexpr idx size() const noexcept { return rows_ * cols_; }

  real *data() { return data_.get(); }
  const real *data() const { return data_.get(); }
  real &operator()(idx i, idx j) { return data_[i * cols_ + j]; }
  real operator()(idx i, idx j) const { return data_[i * cols_ + j]; }

  void to_gpu();
  void to_cpu();
  real *gpu_data() { return d_data_; }
  const real *gpu_data() const { return d_data_; }
  bool on_gpu() const { return d_data_ != nullptr; }

private:
  idx rows_ = 0, cols_ = 0;
  std::unique_ptr<real[]> data_;
  real *d_data_ = nullptr;
};

/// @brief y = A * x
void matvec(const Matrix &A, const Vector &x, Vector &y,
            Backend b = default_backend);

/// @brief C = A * B
void matmul(const Matrix &A, const Matrix &B, Matrix &C,
            Backend b = default_backend);

/// @brief C = alpha*A + beta*B
void matadd(real alpha, const Matrix &A, real beta, const Matrix &B, Matrix &C,
            Backend b = default_backend);

// Named variants for benchmarking and pedagogy.
// Expose the progression of optimisation techniques.
// For production code prefer matmul(A, B, C) or matmul(A, B, C, num::simd).

/// @brief C = A * B  (cache-blocked)
///
/// Divides A, B, C into BLOCKxBLOCK tiles so the working set fits in L2 cache.
/// @param block_size  Tile edge length (default 64).
void matmul_blocked(const Matrix &A, const Matrix &B, Matrix &C,
                    idx block_size = 64);

/// @brief C = A * B  (register-blocked)
///
/// Extends cache blocking with a REGxREG register tile inside each cache tile.
/// @param block_size  Cache tile edge (default 64).
/// @param reg_size    Register tile edge (default 4).
void matmul_register_blocked(const Matrix &A, const Matrix &B, Matrix &C,
                             idx block_size = 64, idx reg_size = 4);

/// @brief C = A * B  (SIMD-accelerated)
///
/// Dispatches at compile time: AVX-256 + FMA on x86, NEON on AArch64,
/// falls back to matmul_blocked if neither is available.
void matmul_simd(const Matrix &A, const Matrix &B, Matrix &C,
                 idx block_size = 64);

/// @brief y = A * x  (SIMD-accelerated)
void matvec_simd(const Matrix &A, const Vector &x, Vector &y);

} // namespace num
