/// @file linear/matrix_properties.hpp
/// @brief Declared mathematical properties and compile-time wrappers for matrices.
#pragma once

#include "kernel/factor.hpp"
#include "core/math/evidence.hpp"
#include "container/concepts.hpp"
#include "container/matrix.hpp"
#include "container/vector.hpp"
#include "linear/debug.hpp"
#include "algebra/properties.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace num {

namespace linear {

/// Maximum absolute difference between mirrored entries of a square matrix.
[[nodiscard]] inline real symmetry_error(const Matrix &A) {
    if (A.rows() != A.cols()) {
        throw std::invalid_argument("symmetry_error: matrix must be square");
    }
    real error = 0.0;
    for (idx row = 0; row < A.rows(); ++row) {
        for (idx column = 0; column < row; ++column) {
            error = std::max(error, std::abs(A(row, column) - A(column, row)));
        }
    }
    return error;
}

/// Maximum mirrored-entry error relative to the largest off-diagonal entry.
[[nodiscard]] inline real relative_symmetry_error(const Matrix &A) {
    if (A.rows() != A.cols()) {
        throw std::invalid_argument("relative_symmetry_error: matrix must be square");
    }
    real error = 0.0;
    real scale = 1.0;
    for (idx row = 0; row < A.rows(); ++row) {
        for (idx column = 0; column < row; ++column) {
            error = std::max(error, std::abs(A(row, column) - A(column, row)));
            scale = std::max(scale, std::abs(A(row, column)));
            scale = std::max(scale, std::abs(A(column, row)));
        }
    }
    return error / scale;
}

/// Test absolute entrywise symmetry using the supplied tolerance.
[[nodiscard]] inline bool is_symmetric(const Matrix &A, real tol = 1e-12) {
    if (A.rows() != A.cols()) {
        return false;
    }
    const idx n = A.rows();
    for (idx i = 0; i < n; ++i) {
        for (idx j = 0; j < i; ++j) {
            if (std::abs(A(i, j) - A(j, i)) > tol) {
                return false;
            }
        }
    }
    return true;
}

/// Test symmetry and positive definiteness.
///
/// A successful Cholesky factorization is the definitive test: it fails exactly
/// when a pivot is not positive. The raw kernel is used rather than `num::cholesky`
/// because that routine requires the very invariant being tested.
[[nodiscard]] inline bool is_spd(const Matrix &A, real tol = 1e-12) {
    if (!is_symmetric(A, tol)) {
        return false;
    }
    Matrix factor(A.rows(), A.cols(), 0.0);
    return kernel::raw::cholesky(factor.data(), A.data(), A.rows());
}

namespace props_detail {

/// Presents a stored matrix through the operator interface the axiom samplers use,
/// so a matrix assertion is checked by exactly the same probes as an operator one.
template <class Mat>
struct MatrixAsOperator {
    const Mat &A;

    [[nodiscard]] idx rows() const noexcept { return A.rows(); }
    [[nodiscard]] idx cols() const noexcept { return A.cols(); }

    template <class X, class Y>
    void apply(const X &x, Y &y) const {
        using T = entry_t<Mat>;
        for (idx i = 0; i < A.rows(); ++i) {
            T sum = T(0);
            for (idx j = 0; j < A.cols(); ++j) {
                sum += A(i, j) * x[j];
            }
            y[i] = sum;
        }
    }
};

/// A matrix carrying any property is square, since every property in the
/// hierarchy is a statement about an endomorphism. Squareness itself is shape
/// rather than a property, so it is carried by a tag instead.
template <class P>
struct square_tag {
    using square_matrix_tag = void;
};

} // namespace props_detail

/// @brief A stored matrix carrying an asserted axiom.
///
/// The matrix and operator sides of the library now record properties in one
/// vocabulary: this and `num::operators::StructuredOp` place their argument at the
/// same position in the `num::property` lattice. A self-adjoint matrix is square by
/// construction, so no separate shape tag is needed here.
template <class Mat, class Ax>
class StructuredMatrix final : public props_detail::square_tag<Ax> {
  public:
    /// @brief Position of this matrix in the property hierarchy.
    using properties = Ax;

    explicit StructuredMatrix(
        Mat A, math::EvidenceProvenance provenance =
                   {math::evidence_origin::assumed, std::source_location::current(),
                    "legacy direct assertion"})
        : A_(std::move(A)), provenance_(provenance) {}

    [[nodiscard]] const Mat &base() const noexcept { return A_; }
    [[nodiscard]] idx rows() const noexcept { return A_.rows(); }
    [[nodiscard]] idx cols() const noexcept { return A_.cols(); }
    [[nodiscard]] entry_t<Mat> operator()(idx i, idx j) const { return A_(i, j); }
    [[nodiscard]] const math::EvidenceProvenance &provenance() const noexcept {
        return provenance_;
    }

  private:
    Mat A_;
    math::EvidenceProvenance provenance_;
};

/// @brief Matrix carrying a caller-provided square dimension guarantee.
///
/// Squareness is shape, not an axiom: it is decidable from the object, so it is
/// deliberately kept out of the property hierarchy.
template <class Mat = Matrix>
class SquareMatrix final {
  public:
    using square_matrix_tag = void;

    explicit SquareMatrix(Mat A) : A_(std::move(A)) {}

    [[nodiscard]] const Mat &base() const noexcept { return A_; }
    [[nodiscard]] idx rows() const noexcept { return A_.rows(); }
    [[nodiscard]] idx cols() const noexcept { return A_.cols(); }
    [[nodiscard]] entry_t<Mat> operator()(idx i, idx j) const { return A_(i, j); }
    [[nodiscard]] entry_t<Mat> &operator()(idx i, idx j) { return A_(i, j); }

  private:
    Mat A_;
};

/// @brief Matrix asserted self-adjoint: \f$A = A^T\f$ over \f$\mathbb{R}\f$, \f$A = A^*\f$ over \f$\mathbb{C}\f$.
template <class Mat = Matrix>
using SymmetricMatrix = StructuredMatrix<Mat, property::self_adjoint>;

/// @brief Matrix asserted positive semi-definite: \f$x^T A x \geq 0\f$.
template <class Mat = Matrix>
using PSDMatrix = StructuredMatrix<Mat, property::psd>;

/// @brief Matrix asserted positive definite: \f$x^T A x > 0\f$ for \f$x \neq 0\f$.
template <class Mat = Matrix>
using SPDMatrix = StructuredMatrix<Mat, property::spd>;

/// @brief Matrix asserted Hermitian; the same claim as `SymmetricMatrix` over a complex field.
template <class Mat = Matrix>
using HermitianMatrix = StructuredMatrix<Mat, property::self_adjoint>;

/// @brief Attach axiom Ax to a matrix, sampling it and every axiom it implies.
///
/// Unlike the earlier `assume_*`, which attached a tag without looking at the
/// entries, this runs the same probes the operator side uses. Under
/// `preset::production` the checks compile away.
template <class Ax, class Mat>
[[nodiscard]] inline StructuredMatrix<Mat, Ax>
assume_property(Mat A, std::source_location loc = std::source_location::current()) {
    if (A.rows() == A.cols()) {
        const props_detail::MatrixAsOperator<Mat> view{A};
        verify_property<Ax, BasicVector<entry_t<Mat>>>(view, loc);
    }
    return StructuredMatrix<Mat, Ax>(
        std::move(A), {math::evidence_origin::assumed, loc, "legacy sampled assertion"});
}

/// Attach a square matrix guarantee without checking the entries.
template <class Mat = Matrix>
[[nodiscard]] inline SquareMatrix<Mat> assume_square(Mat A) {
    return SquareMatrix<Mat>(std::move(A));
}

/// Attach a symmetry guarantee, sampled under the active diagnostic preset.
template <class Mat = Matrix>
[[nodiscard]] inline SymmetricMatrix<Mat>
assume_symmetric(Mat A, std::source_location loc = std::source_location::current()) {
    return assume_property<property::self_adjoint>(std::move(A), loc);
}

/// Attach a positive semi-definiteness guarantee, sampled under the active preset.
template <class Mat = Matrix>
[[nodiscard]] inline PSDMatrix<Mat>
assume_psd(Mat A, std::source_location loc = std::source_location::current()) {
    return assume_property<property::psd>(std::move(A), loc);
}

/// Attach an SPD guarantee, sampled under the active diagnostic preset.
template <class Mat = Matrix>
[[nodiscard]] inline SPDMatrix<Mat>
assume_spd(Mat A, std::source_location loc = std::source_location::current()) {
    return assume_property<property::spd>(std::move(A), loc);
}

/// Attach a Hermitian guarantee, sampled under the active diagnostic preset.
template <class Mat = Matrix>
[[nodiscard]] inline HermitianMatrix<Mat>
assume_hermitian(Mat A, std::source_location loc = std::source_location::current()) {
    return assume_property<property::self_adjoint>(std::move(A), loc);
}

/// Validate square dimensions before constructing a property wrapper.
template <class Mat = Matrix>
[[nodiscard]] inline SquareMatrix<Mat> make_square(Mat A) {
    if (A.rows() != A.cols()) {
        throw std::invalid_argument("make_square: matrix must be square");
    }
    return SquareMatrix<Mat>(std::move(A));
}

/// Validate symmetry exhaustively before constructing a property wrapper.
template <class Mat = Matrix>
[[nodiscard]] inline SymmetricMatrix<Mat>
make_symmetric(Mat A, real tol = 1e-12,
               std::source_location loc = std::source_location::current()) {
    if (!is_symmetric(A, tol)) {
        throw std::invalid_argument("make_symmetric: matrix is not symmetric");
    }
    return SymmetricMatrix<Mat>(
        std::move(A), {math::evidence_origin::verified, loc,
                       "exhaustive symmetry validator"});
}

/// Validate positive definiteness exhaustively before constructing a property wrapper.
template <class Mat = Matrix>
[[nodiscard]] inline SPDMatrix<Mat>
make_spd(Mat A, real tol = 1e-12,
         std::source_location loc = std::source_location::current()) {
    if (!is_spd(A, tol)) {
        throw std::invalid_argument("make_spd: matrix is not symmetric positive definite");
    }
    return SPDMatrix<Mat>(
        std::move(A), {math::evidence_origin::verified, loc,
                       "exhaustive Cholesky validator"});
}


// -----------------------------------------------------------------------------
// Structural taggers
// -----------------------------------------------------------------------------
//
// Bandedness, tridiagonality and CSR validity are *decidable* facts about stored
// data, not axioms about a linear map. So unlike assume_spd, which can only
// sample, these verify exhaustively before attaching the tag.

/// @brief Matrix carrying an asserted band structure \f$A_{ij} = 0\f$ outside \f$-k_l \leq j-i \leq k_u\f$.
template <class Mat = Matrix>
class BandedMatrixView final {
  public:
    using banded_matrix_tag = void;

    BandedMatrixView(Mat A, idx lower, idx upper)
        : A_(std::move(A)), kl_(lower), ku_(upper) {}

    [[nodiscard]] const Mat &base() const noexcept { return A_; }
    [[nodiscard]] idx rows() const noexcept { return A_.rows(); }
    [[nodiscard]] idx cols() const noexcept { return A_.cols(); }
    [[nodiscard]] idx kl() const noexcept { return kl_; }
    [[nodiscard]] idx ku() const noexcept { return ku_; }
    [[nodiscard]] entry_t<Mat> operator()(idx i, idx j) const { return A_(i, j); }

  private:
    Mat A_;
    idx kl_;
    idx ku_;
};

/// @brief Tridiagonal system held as its three occupied diagonals.
template <class Vec = Vector>
class TridiagonalMatrix final {
  public:
    using tridiagonal_matrix_tag = void;

    TridiagonalMatrix(Vec sub, Vec main, Vec super)
        : dl(std::move(sub)), d(std::move(main)), du(std::move(super)) {}

    Vec dl; ///< Subdiagonal, length n-1.
    Vec d;  ///< Main diagonal, length n.
    Vec du; ///< Superdiagonal, length n-1.

    [[nodiscard]] idx size() const noexcept { return d.size(); }
    [[nodiscard]] idx rows() const noexcept { return d.size(); }
    [[nodiscard]] idx cols() const noexcept { return d.size(); }
};

/// @brief Sparse matrix carrying a verified CSR structural guarantee.
template <class Mat>
class SparseCSRMatrix final {
  public:
    using sparse_csr_matrix_tag = void;

    explicit SparseCSRMatrix(Mat A) : A_(std::move(A)) {}

    [[nodiscard]] const Mat &base() const noexcept { return A_; }
    [[nodiscard]] idx n_rows() const { return A_.n_rows(); }
    [[nodiscard]] idx n_cols() const { return A_.n_cols(); }
    [[nodiscard]] idx nnz() const { return A_.nnz(); }
    [[nodiscard]] auto row_ptr() const { return A_.row_ptr(); }
    [[nodiscard]] auto col_idx() const { return A_.col_idx(); }
    [[nodiscard]] auto values() const { return A_.values(); }

  private:
    Mat A_;
};

/// @brief Assert a band structure, checking that entries outside the band vanish.
template <class Mat = Matrix>
[[nodiscard]] inline BandedMatrixView<Mat>
assume_banded(Mat A, idx lower, idx upper,
              std::source_location loc = std::source_location::current()) {
    BandedMatrixView<Mat> tagged(std::move(A), lower, upper);
    debug::verify_banded_structure(tagged, loc);
    debug::verify_band_occupancy(tagged, 0.0, loc);
    return tagged;
}

/// @brief Assert a tridiagonal structure, checking the three diagonals are consistently sized.
template <class Vec = Vector>
[[nodiscard]] inline TridiagonalMatrix<Vec>
assume_tridiagonal(Vec sub, Vec main, Vec super,
                   std::source_location loc = std::source_location::current()) {
    debug::verify_tridiagonal_structure(sub, main, super, loc);
    return TridiagonalMatrix<Vec>(std::move(sub), std::move(main), std::move(super));
}

/// @brief Assert valid CSR storage: monotonic row offsets, in-range columns, finite values.
template <class Mat>
[[nodiscard]] inline SparseCSRMatrix<Mat>
assume_sparse_csr(Mat A, std::source_location loc = std::source_location::current()) {
    debug::verify_sparse_structure(A, loc);
    return SparseCSRMatrix<Mat>(std::move(A));
}

} // namespace linear

// Expose property types and assume_* / make_* taggers in top-level num:: namespace
using linear::PSDMatrix;
using linear::SPDMatrix;
using linear::SquareMatrix;
using linear::StructuredMatrix;
using linear::SymmetricMatrix;
using linear::HermitianMatrix;

using linear::assume_hermitian;
using linear::assume_property;
using linear::assume_psd;
using linear::assume_spd;
using linear::assume_square;
using linear::assume_symmetric;

using linear::assume_banded;
using linear::assume_sparse_csr;
using linear::assume_tridiagonal;
using linear::BandedMatrixView;
using linear::SparseCSRMatrix;
using linear::TridiagonalMatrix;

using linear::make_spd;
using linear::make_square;
using linear::make_symmetric;

} // namespace num
