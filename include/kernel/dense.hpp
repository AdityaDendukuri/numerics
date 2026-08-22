/// @file kernel/dense.hpp
/// @brief Dense matrix inner kernels  (namespace num::kernel::dense)
#pragma once

#include "core/matrix.hpp"
#include "core/types.hpp"
#include "core/vector.hpp"
#include "kernel/policy.hpp"

namespace num::kernel::dense {

/// @brief Sequential rank-1 update.
void ger(real alpha, const Vector &x, const Vector &y, Matrix &A, seq_t) noexcept;

/// @brief Parallel rank-1 update.
void ger(real alpha, const Vector &x, const Vector &y, Matrix &A, par_t);

inline void ger(real alpha, const Vector &x, const Vector &y, Matrix &A) {
    ger(alpha, x, y, A, default_policy{});
}

/// @brief Forward substitution: solve Lx = b.
void trsv_lower(const Matrix &L, const Vector &b, Vector &x);

/// @brief Back substitution: solve Ux = b.
void trsv_upper(const Matrix &U, const Vector &b, Vector &x);

} // namespace num::kernel::dense
