/// @file linear/solvers/gmres.hpp
/// @brief Evidence-constrained restarted GMRES.
#pragma once

#include "core/policy.hpp"
#include "linear/math_adapters.hpp"
#include "linear/solvers/math_gmres.hpp"
#include "linear/sparse/sparse_op.hpp"
#include "operator/dense.hpp"

namespace num {

/// Compatibility spelling for operator callers passing scalar options.
template <class Op>
requires math::InnerProductSpace<Vector> &&math::EndomorphismOn<Op, Vector> inline SolverResult
gmres(const Op &A, const Vector &b, Vector &x, real tolerance, idx max_iterations = 1000,
      idx restart = 30) {
    return gmres(
        A, b, x,
        GMRESOptions{.tolerance = tolerance, .max_iterations = max_iterations, .restart = restart});
}

/// Stored sparse system adapter.
inline SolverResult gmres(const SparseMatrix &A, const Vector &b, Vector &x, real tolerance = 1e-6,
                          idx max_iterations = 1000, idx restart = 30) {
    if (A.n_rows() != A.n_cols()) {
        throw std::invalid_argument("gmres: sparse matrix must be square");
    }
    return gmres(
        operators::SparseOp(A), b, x,
        GMRESOptions{.tolerance = tolerance, .max_iterations = max_iterations, .restart = restart});
}

/// Stored dense system adapter. The selected backend is localized to matvec.
inline SolverResult gmres(const Matrix &A, const Vector &b, Vector &x, real tolerance = 1e-6,
                          idx max_iterations = 1000, idx restart = 30,
                          Backend selected_backend = backend::dflt) {
    if (A.rows() != A.cols()) {
        throw std::invalid_argument("gmres: dense matrix must be square");
    }
    return gmres(
        operators::DenseOp(A, selected_backend), b, x,
        GMRESOptions{.tolerance = tolerance, .max_iterations = max_iterations, .restart = restart});
}

} // namespace num
