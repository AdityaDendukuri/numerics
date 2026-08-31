/// @file linear/concepts.hpp
/// @brief Structural and property contracts for linear algebra objects and solvers.
///
/// Matrices and matrix-free operators state their properties in one vocabulary:
/// both record a position in the `num::property` lattice, so `SPDMatrixLike` and
/// `num::SPDOperator` mean the same mathematical thing about different
/// representations.
///
/// Storage formats — CSR, banded, tridiagonal — are not properties of a linear map
/// and live in `num::repr`. A routine requiring one of those is making a
/// statement about memory, and should say so.
#pragma once

#include "container/concepts.hpp"
#include "core/types.hpp"
#include "algebra/properties.hpp"
#include "linear/matrix_properties.hpp"
#include "linear/solvers/solver_result.hpp"
#include "operator/concepts.hpp"
#include <concepts>
#include <type_traits>
#include <vector>

namespace num {

namespace detail {

template <class M>
concept has_square_tag = requires {
    typename std::remove_cvref_t<M>::square_matrix_tag;
};

} // namespace detail

// =============================================================================
// 1. Matrix invariants
// =============================================================================

/// @brief Matrix carrying a square-dimension guarantee \f$A: V \to V\f$.
///
/// Squareness of a dynamically sized matrix is a runtime fact, so this asks
/// whether a guarantee has been *attached* — by `assume_square`, `make_square`, or
/// by a property that implies it. It deliberately does not ask whether
/// `rows() == cols()` is a well-formed expression, which is true of every matrix
/// and therefore says nothing.
template <class M>
concept SquareMatrixLike =
    MatrixSpace<M> && (detail::has_square_tag<M> || Asserts<M, property::self_adjoint>);

/// @brief Matrix asserted self-adjoint: \f$A = A^T\f$, or \f$A = A^*\f$ over \f$\mathbb{C}\f$.
template <class M>
concept SymmetricMatrixLike =
    SquareMatrixLike<M> && Asserts<M, property::self_adjoint>;

/// @brief Matrix asserted positive semi-definite: \f$x^T A x \geq 0\f$; may be singular.
template <class M>
concept PSDMatrixLike =
    SymmetricMatrixLike<M> && Asserts<M, property::psd>;

/// @brief Matrix asserted positive definite: \f$x^T A x > 0\f$ for \f$x \neq 0\f$; admits a Cholesky factor.
template <class M>
concept SPDMatrixLike =
    PSDMatrixLike<M> && Asserts<M, property::spd>;

// =============================================================================
// 2. Storage formats
// =============================================================================
//
// Aliases into num::repr. A banded solver needs banded *storage*; bandedness is
// not a property of the underlying linear map.

/// @brief Banded storage with \f$A_{ij} = 0\f$ outside \f$-k_l \leq j-i \leq k_u\f$.
template <class B>
concept BandedMatrixLike = repr::Banded<B>;

/// @brief Tridiagonal storage exposing subdiagonal, diagonal and superdiagonal.
template <class T>
concept TridiagonalMatrixLike = repr::Tridiagonal<T>;

/// @brief Compressed sparse row storage.
template <class M>
concept SparseMatrixCSRLike = repr::CSR<M>;

// =============================================================================
// 3. Direct factorizations
// =============================================================================

/// @brief Factor admitting a substitution solve for a single right-hand side.
///
/// The library's factorizations are plain results paired with free functions, so
/// the contract is `solve_in_place(factor, rhs)` found by argument-dependent
/// lookup. Requiring a `.solve()` member, as this once did, made the concept
/// unsatisfiable by `CholeskyResult` and `LUResult` alike.
template <class F, class Vec = Vector>
concept TriangularFactor = VectorSpace<Vec> && requires(const F &factor, Vec &rhs) {
    solve_in_place(factor, rhs);
};

/// @brief Reusable factorization of \f$A\f$ solving single and multiple right-hand sides.
///
/// The defining property is reuse: the decomposition is computed once and applied
/// to any number of right-hand sides, which is what distinguishes a factorization
/// from a one-shot solve. That is expressed by also accepting a matrix of
/// right-hand sides.
template <class F, class Vec = Vector, class Mat = Matrix>
concept DirectFactorization = TriangularFactor<F, Vec> && requires(const F &factor, Mat &rhs) {
    solve_in_place(factor, rhs);
};

// =============================================================================
// 4. Solvers and preconditioners
// =============================================================================

/// @brief Approximate inverse \f$z \approx M^{-1} r\f$ applied once per iteration.
///
/// Structurally a linear operator; what makes it a preconditioner is intent, not
/// interface, so it is stated as such rather than given a distinguishing method.
template <class M, class X = Vector, class Y = Vector>
concept Preconditioner = LinearOperator<M, X, Y>;

/// @brief Preconditioner asserted self-adjoint, as PCG and MINRES require.
///
/// A non-symmetric preconditioner silently destroys the Krylov space those methods
/// build; this is the constraint that should gate them.
template <class M, class X = Vector, class Y = Vector>
concept SymmetricPreconditioner = Preconditioner<M, X, Y> && SelfAdjointOperator<M, X, Y>;

/// @brief Preconditioner asserted SPD, the precondition for PCG's error norm to be a norm.
template <class M, class X = Vector, class Y = Vector>
concept SPDPreconditioner = SymmetricPreconditioner<M, X, Y> && SPDOperator<M, X, Y>;

// =============================================================================
// 5. Matrix functions
// =============================================================================


} // namespace num
