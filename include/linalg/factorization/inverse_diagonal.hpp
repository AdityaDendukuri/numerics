/// @file linalg/factorization/inverse_diagonal.hpp
/// @brief Diagonal of an inverse from reusable factorizations.
#pragma once

#include "linalg/factorization/cholesky.hpp"
#include "linalg/factorization/lu.hpp"
#include "linalg/solvers/auto_linear.hpp"
#include "linalg/sparse/klu.hpp"
#include <span>

namespace num {

struct InverseDiagonalWorkspace {
  Matrix right_hand_sides;
  Matrix solutions;
};

void inverse_diagonal(const LUResult& factor,
                      Vector& result,
                      InverseDiagonalWorkspace& workspace,
                      idx block_size = 64);
void inverse_diagonal(const CholeskyResult& factor,
                      Vector& result,
                      InverseDiagonalWorkspace& workspace,
                      idx block_size = 64);
void inverse_diagonal(const KLUFactor& factor,
                      Vector& result,
                      InverseDiagonalWorkspace& workspace,
                      idx block_size = 64);
void inverse_diagonal(const AutoLinearSolver& factor,
                      Vector& result,
                      InverseDiagonalWorkspace& workspace,
                      idx block_size = 64);

void selected_inverse(const LUResult& factor,
                      std::span<const idx> rows,
                      std::span<const idx> columns,
                      Vector& result,
                      InverseDiagonalWorkspace& workspace);
void selected_inverse(const CholeskyResult& factor,
                      std::span<const idx> rows,
                      std::span<const idx> columns,
                      Vector& result,
                      InverseDiagonalWorkspace& workspace);
void selected_inverse(const KLUFactor& factor,
                      std::span<const idx> rows,
                      std::span<const idx> columns,
                      Vector& result,
                      InverseDiagonalWorkspace& workspace);
void selected_inverse(const AutoLinearSolver& factor,
                      std::span<const idx> rows,
                      std::span<const idx> columns,
                      Vector& result,
                      InverseDiagonalWorkspace& workspace);

} // namespace num
