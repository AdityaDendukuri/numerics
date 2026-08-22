#include "linalg/factorization/inverse_diagonal.hpp"
#include "linalg/matrix_properties.hpp"
#include "linalg/matrix_utils.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace num {
namespace {

constexpr real residual_tolerance = 1e-10;

void validate_update(idx size, idx updated_size, const Vector &diagonal, const Matrix &vectors,
                     const Matrix &weights, const Vector &result, idx fallback_block_size) {
    const idx rank = vectors.cols();
    if (updated_size != size || diagonal.size() != size || result.size() != size ||
        vectors.rows() != size || weights.rows() != rank || weights.cols() != rank ||
        fallback_block_size == 0) {
        throw std::invalid_argument("inverse_diagonal_after_update: dimension mismatch");
    }
}

[[nodiscard]] Matrix product(const Matrix &left, const Matrix &right) {
    Matrix result(left.rows(), right.cols(), 0.0);
    matmul(left, right, result);
    return result;
}

void add_identity(Matrix &matrix) {
    for (idx index = 0; index < matrix.rows(); ++index) {
        matrix(index, index) += 1.0;
    }
}

void symmetrize(Matrix &matrix) {
    for (idx row = 0; row < matrix.rows(); ++row) {
        for (idx column = 0; column < row; ++column) {
            const real average = 0.5 * (matrix(row, column) + matrix(column, row));
            matrix(row, column) = average;
            matrix(column, row) = average;
        }
    }
}

[[nodiscard]] bool unsafe_subtraction(real before, real correction, real after) {
    if (!std::isfinite(after)) {
        return true;
    }
    const real scale = std::abs(before) + std::abs(correction);
    return before > 0.0 && after <= std::sqrt(std::numeric_limits<real>::epsilon()) * scale;
}

template <typename Direct>
InverseDiagonalUpdatePath use_direct(Direct &&direct) {
    direct();
    return InverseDiagonalUpdatePath::direct;
}

template <typename Direct>
InverseDiagonalUpdatePath subtract_corrections(const Vector &diagonal, const Vector &corrections,
                                               Vector &result, Direct &&direct) {
    for (idx state = 0; state < diagonal.size(); ++state) {
        result[state] = diagonal[state] - corrections[state];
        if (unsafe_subtraction(diagonal[state], corrections[state], result[state])) {
            return use_direct(direct);
        }
    }
    return InverseDiagonalUpdatePath::woodbury;
}

[[nodiscard]] Vector symmetric_corrections(const Matrix &inverse_vectors,
                                           const CholeskyResult &reduced_factor) {
    const idx size = inverse_vectors.rows();
    const idx rank = inverse_vectors.cols();
    Vector corrections(size, 0.0);
    Vector transformed(rank, 0.0);
    for (idx state = 0; state < size; ++state) {
        for (idx row = 0; row < rank; ++row) {
            real value = inverse_vectors(state, row);
            for (idx inner = 0; inner < row; ++inner) {
                value -= reduced_factor.L(row, inner) * transformed[inner];
            }
            transformed[row] = value / reduced_factor.L(row, row);
            corrections[state] += transformed[row] * transformed[row];
        }
    }
    return corrections;
}

[[nodiscard]] Vector general_corrections(const Matrix &left, const Matrix &reduced_solution) {
    Vector corrections(left.rows(), 0.0);
    for (idx state = 0; state < left.rows(); ++state) {
        for (idx update = 0; update < left.cols(); ++update) {
            corrections[state] += left(state, update) * reduced_solution(update, state);
        }
    }
    return corrections;
}

[[nodiscard]] bool solve_is_accurate(const Matrix &matrix, const Matrix &solution,
                                     const Matrix &right_hand_side) {
    const Matrix reconstructed = product(matrix, solution);
    real error = 0.0;
    real scale = 1.0;
    for (idx row = 0; row < reconstructed.rows(); ++row) {
        for (idx column = 0; column < reconstructed.cols(); ++column) {
            error = std::max(error,
                             std::abs(reconstructed(row, column) - right_hand_side(row, column)));
            scale = std::max(scale, std::abs(right_hand_side(row, column)));
        }
    }
    return error <= residual_tolerance * scale;
}

template <typename Solve, typename Direct>
InverseDiagonalUpdatePath symmetric_woodbury(idx size, idx updated_size, const Vector &diagonal,
                                             const Matrix &vectors, const Matrix &weights,
                                             Vector &result, idx fallback_block_size, Solve &&solve,
                                             Direct &&direct) {
    validate_update(size, updated_size, diagonal, vectors, weights, result, fallback_block_size);
    if (vectors.cols() == 0) {
        result = diagonal;
        return InverseDiagonalUpdatePath::woodbury;
    }

    // Write W=LL^T so the update becomes (UL)(UL)^T.
    if (!linalg::is_symmetric(weights, 64.0 * std::numeric_limits<real>::epsilon())) {
        return use_direct(direct);
    }
    const CholeskyResult weight_factor = cholesky(weights);
    if (!weight_factor.success) {
        return use_direct(direct);
    }
    const Matrix scaled_vectors = product(vectors, weight_factor.L);

    Matrix inverse_vectors(size, vectors.cols(), 0.0);
    solve(scaled_vectors, inverse_vectors);

    // C C^T = I + (UL)^T A^-1 (UL).
    Matrix reduced = product(transpose(scaled_vectors), inverse_vectors);
    symmetrize(reduced);
    add_identity(reduced);
    const CholeskyResult reduced_factor = cholesky(reduced);
    if (!reduced_factor.success) {
        return use_direct(direct);
    }

    return subtract_corrections(diagonal, symmetric_corrections(inverse_vectors, reduced_factor),
                                result, direct);
}

template <typename Solve, typename SolveTranspose, typename Direct>
InverseDiagonalUpdatePath general_woodbury(idx size, idx updated_size, const Vector &diagonal,
                                           const Matrix &vectors, const Matrix &weights,
                                           Vector &result, idx fallback_block_size, Solve &&solve,
                                           SolveTranspose &&solve_transpose, Direct &&direct) {
    validate_update(size, updated_size, diagonal, vectors, weights, result, fallback_block_size);
    if (vectors.cols() == 0) {
        result = diagonal;
        return InverseDiagonalUpdatePath::woodbury;
    }

    Matrix left(size, vectors.cols(), 0.0);
    solve(vectors, left);
    Matrix right_transpose(size, vectors.cols(), 0.0);
    solve_transpose(vectors, right_transpose);

    // Solve (I + W U^T A^-1 U)Y = W U^T A^-1.
    const Matrix gram = product(transpose(vectors), left);
    Matrix reduced = product(weights, gram);
    add_identity(reduced);
    const LUResult reduced_factor = lu(reduced);
    if (reduced_factor.singular) {
        return use_direct(direct);
    }
    const Matrix reduced_right_hand_side = product(weights, transpose(right_transpose));
    Matrix reduced_solution(vectors.cols(), size, 0.0);
    lu_solve(reduced_factor, reduced_right_hand_side, reduced_solution);
    if (!solve_is_accurate(reduced, reduced_solution, reduced_right_hand_side)) {
        return use_direct(direct);
    }

    return subtract_corrections(diagonal, general_corrections(left, reduced_solution), result,
                                direct);
}

} // namespace

InverseDiagonalUpdatePath
inverse_diagonal_after_update(const LUResult &factor, const LUResult &updated_factor,
                              const Vector &diagonal, const Matrix &update_vectors,
                              const Matrix &update_weights, Vector &result,
                              InverseDiagonalWorkspace &workspace, idx fallback_block_size) {
    if (factor.singular || updated_factor.singular) {
        throw std::invalid_argument("inverse_diagonal_after_update: factors must be nonsingular");
    }
    return general_woodbury(
        factor.LU.rows(), updated_factor.LU.rows(), diagonal, update_vectors, update_weights,
        result, fallback_block_size,
        [&](const Matrix &rhs, Matrix &solution) { lu_solve(factor, rhs, solution); },
        [&](const Matrix &rhs, Matrix &solution) { lu_solve_transpose(factor, rhs, solution); },
        [&] { inverse_diagonal(updated_factor, result, workspace, fallback_block_size); });
}

InverseDiagonalUpdatePath
inverse_diagonal_after_update(const CholeskyResult &factor, const CholeskyResult &updated_factor,
                              const Vector &diagonal, const Matrix &update_vectors,
                              const Matrix &update_weights, Vector &result,
                              InverseDiagonalWorkspace &workspace, idx fallback_block_size) {
    if (!factor.success || !updated_factor.success) {
        throw std::invalid_argument(
            "inverse_diagonal_after_update: factors must be positive definite");
    }
    return symmetric_woodbury(
        factor.L.rows(), updated_factor.L.rows(), diagonal, update_vectors, update_weights, result,
        fallback_block_size,
        [&](const Matrix &rhs, Matrix &solution) { cholesky_solve(factor, rhs, solution); },
        [&] { inverse_diagonal(updated_factor, result, workspace, fallback_block_size); });
}

InverseDiagonalUpdatePath
inverse_diagonal_after_update(const KLUFactor &factor, const KLUFactor &updated_factor,
                              const Vector &diagonal, const Matrix &update_vectors,
                              const Matrix &update_weights, Vector &result,
                              InverseDiagonalWorkspace &workspace, idx fallback_block_size) {
    return general_woodbury(
        factor.size(), updated_factor.size(), diagonal, update_vectors, update_weights, result,
        fallback_block_size,
        [&](const Matrix &rhs, Matrix &solution) { factor.solve(rhs, solution); },
        [&](const Matrix &rhs, Matrix &solution) { factor.solve_transpose(rhs, solution); },
        [&] { inverse_diagonal(updated_factor, result, workspace, fallback_block_size); });
}

InverseDiagonalUpdatePath inverse_diagonal_after_update(
    const AutoLinearSolver &factor, const AutoLinearSolver &updated_factor, const Vector &diagonal,
    const Matrix &update_vectors, const Matrix &update_weights, Vector &result,
    InverseDiagonalWorkspace &workspace, idx fallback_block_size) {
    return general_woodbury(
        factor.size(), updated_factor.size(), diagonal, update_vectors, update_weights, result,
        fallback_block_size,
        [&](const Matrix &rhs, Matrix &solution) { factor.solve(rhs, solution); },
        [&](const Matrix &rhs, Matrix &solution) { factor.solve_transpose(rhs, solution); },
        [&] { inverse_diagonal(updated_factor, result, workspace, fallback_block_size); });
}

} // namespace num
