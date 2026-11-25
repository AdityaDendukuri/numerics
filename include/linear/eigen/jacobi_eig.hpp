/// @file linear/eigen/jacobi_eig.hpp
/// @brief Symmetric eigendecomposition by cyclic Jacobi.
#pragma once

#include "kernel/kernel.hpp"
#include "linear/matrix_properties.hpp"
#include "linear/solvers/solver_result.hpp"
#include "operator/concepts.hpp"

#include "container/matrix.hpp"
#include "core/policy.hpp"
#include "container/vector.hpp"

#include <ostream>

namespace num {

// Jacobi Rotations

/// @brief Jacobi rotation: \f$(c, s)\f$ zeroing \f$a_{pq}\f$ of a symmetric \f$2 \times 2\f$ block.
template <std::floating_point T>
NUM_K_AINLINE void jacobi_rotation(T app, T aqq, T apq, T &c, T &s) noexcept {
    if (std::abs(apq) < T(1e-15)) {
        c = T(1);
        s = T(0);
        return;
    }
    const T tau = (aqq - app) / (T(2) * apq);
    const T t = std::copysign(T(1), tau) / (std::abs(tau) + std::sqrt(T(1) + (tau * tau)));
    c = T(1) / std::sqrt(T(1) + (t * t));
    s = c * t;
}

} // namespace num

namespace num {

/// @brief Symmetric eigendecomposition result: \f$A = V \Lambda V^T\f$.
struct eigen_result {
    vec<real> values;          ///< Eigenvalues in ascending order.
    mat<real> vectors;         ///< Corresponding eigenvectors stored as columns.
    idx sweeps = 0;         ///< Jacobi sweeps for the fallback implementation.
    bool converged = false; ///< Whether the requested tolerance was met.

    friend std::ostream &operator<<(std::ostream &os, const eigen_result &r) {
        os << "eigen_result{ values: [" << r.values.size() << " eigenvalues]"
           << ", vectors: " << r.vectors.rows() << "x" << r.vectors.cols()
           << ", sweeps: " << r.sweeps
           << ", converged: " << (r.converged ? "true" : "false") << " }";
        return os;
    }
};

/// @brief Symmetric eigendecomposition: \f$A = V \Lambda V^T\f$, eigenvalues sorted.
/// Uses LAPACK `dsyevd` when available, else cyclic Jacobi.
eigen_result eig_sym(const with_law<mat<real>, law::self_adjoint> &A, real tol = 1e-12, idx max_sweeps = 100);

namespace seq {
eigen_result eig_sym(const mat<real> &A, real tol, idx max_sweeps);
} // namespace seq
namespace omp {
eigen_result eig_sym(const mat<real> &A, real tol, idx max_sweeps);
} // namespace omp
namespace lapack {
eigen_result eig_sym(const mat<real> &A);
} // namespace lapack

namespace unsafe {

/// @brief Symmetric eigendecomposition, unchecked: \f$A = V \Lambda V^T\f$.
/// Reads the lower triangle.
/// @warning An asymmetric matrix yields the spectrum of its symmetric part, not an error.
eigen_result eig_sym(const mat<real> &A, real tol = 1e-12, idx max_sweeps = 100);

} // namespace unsafe

/// @brief Compile-time rejection of an untagged matrix; tag it symmetric first.
template <class M>
requires matrix_space<M> && (!claims<M, law::self_adjoint>)
eigen_result eig_sym(const M & /*untagged*/, real = 1e-12, idx = 100) {
    static_assert(claims<M, law::self_adjoint>,
                  "eig_sym() requires a matrix carrying the symmetry invariant, which is what "
                  "guarantees a real spectrum and an orthogonal eigenbasis. "
                  "Establish it with num::assume_symmetric(A) or num::make_symmetric(A). "
                  "To bypass the invariant deliberately, call num::unsafe::eig_sym(A).");
    return {};
}

} // namespace num
