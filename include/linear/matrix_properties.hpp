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
[[nodiscard]] inline real symmetry_error(const mat &A) {
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
[[nodiscard]] inline real relative_symmetry_error(const mat &A) {
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
[[nodiscard]] inline bool is_symmetric(const mat &A, real tol = 1e-12) {
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
[[nodiscard]] inline bool is_spd(const mat &A, real tol = 1e-12) {
    if (!is_symmetric(A, tol)) {
        return false;
    }
    mat factor(A.rows(), A.cols(), 0.0);
    return kernel::cholesky(factor.data(), A.data(), A.rows());
}

namespace props_detail {

/// Presents a stored matrix through the operator interface the axiom samplers use,
/// so a matrix assertion is checked by exactly the same probes as an operator one.
template <class Mat>
struct matrix_as_operator {
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
/// vocabulary: this and `num::operators::structured_op` place their argument at the
/// same position in the `num::property` lattice. A self-adjoint matrix is square by
/// construction, so no separate shape tag is needed here.
template <class Mat, class Ax>
class structured_mat final : public props_detail::square_tag<Ax> {
  public:
    using domain_type = vec;
    using codomain_type = vec;
    using math_laws = std::conditional_t<
        std::derived_from<Ax, law::spd>, math::type_list<law::spd>,
        std::conditional_t<
            std::derived_from<Ax, law::psd>, math::type_list<law::psd>,
            std::conditional_t<std::derived_from<Ax, law::self_adjoint>,
                               math::type_list<law::self_adjoint>,
                               math::type_list<law::linear_map>>>>;

    explicit structured_mat(
        Mat A, math::evidence_provenance provenance =
                   {math::evidence_origin::assumed, std::source_location::current(),
                    "legacy direct assertion"})
        : A_(std::move(A)), provenance_(provenance) {}

    [[nodiscard]] const Mat &base() const noexcept { return A_; }
    [[nodiscard]] idx rows() const noexcept { return A_.rows(); }
    [[nodiscard]] idx cols() const noexcept { return A_.cols(); }
    [[nodiscard]] entry_t<Mat> operator()(idx i, idx j) const { return A_(i, j); }
    [[nodiscard]] const math::evidence_provenance &provenance() const noexcept {
        return provenance_;
    }

    template <class X = vec, class Y = vec>
    void apply(const X &x, Y &y) const {
        if constexpr (requires(const Mat &m, const X &in, Y &out) { m.apply(in, out); }) {
            A_.apply(x, y);
        } else {
            props_detail::matrix_as_operator<Mat>{A_}.apply(x, y);
        }
    }

  private:
    Mat A_;
    math::evidence_provenance provenance_;
};

/// @brief mat carrying a caller-provided square dimension guarantee.
///
/// Squareness is shape, not an axiom: it is decidable from the object, so it is
/// deliberately kept out of the property hierarchy.
template <class Mat = mat>
class sq_mat final {
  public:
    using square_matrix_tag = void;
    using domain_type = vec;
    using codomain_type = vec;

    explicit sq_mat(Mat A) : A_(std::move(A)) {}

    [[nodiscard]] const Mat &base() const noexcept { return A_; }
    [[nodiscard]] idx rows() const noexcept { return A_.rows(); }
    [[nodiscard]] idx cols() const noexcept { return A_.cols(); }
    [[nodiscard]] entry_t<Mat> operator()(idx i, idx j) const { return A_(i, j); }
    [[nodiscard]] entry_t<Mat> &operator()(idx i, idx j) { return A_(i, j); }

    template <class X = vec, class Y = vec>
    void apply(const X &x, Y &y) const {
        if constexpr (requires(const Mat &m, const X &in, Y &out) { m.apply(in, out); }) {
            A_.apply(x, y);
        } else {
            props_detail::matrix_as_operator<Mat>{A_}.apply(x, y);
        }
    }

  private:
    Mat A_;
};

/// @brief mat asserted self-adjoint: \f$A = A^T\f$ over \f$\mathbb{R}\f$, \f$A = A^*\f$ over \f$\mathbb{C}\f$.
template <class Mat = mat>
using sym_mat = structured_mat<Mat, law::self_adjoint>;

/// @brief mat asserted positive semi-definite: \f$x^T A x \geq 0\f$.
template <class Mat = mat>
using psd_matrix = structured_mat<Mat, law::psd>;

/// @brief mat asserted positive definite: \f$x^T A x > 0\f$ for \f$x \neq 0\f$.
template <class Mat = mat>
using spd_mat = structured_mat<Mat, law::spd>;

/// @brief mat asserted Hermitian; the same claim as `sym_mat` over a complex field.
template <class Mat = mat>
using hermitian_matrix = structured_mat<Mat, law::self_adjoint>;

/// @brief Attach axiom Ax to a matrix, sampling it and every axiom it implies.
///
/// Unlike the earlier `assume_*`, which attached a tag without looking at the
/// entries, this runs the same probes the operator side uses. Under
/// `preset::production` the checks compile away.
/// @brief Attach axiom Ax to a matrix, sampling it and every axiom it implies.
///
/// Unlike the earlier `assume_*`, which attached a tag without looking at the
/// entries, this runs the same probes the operator side uses. Under
/// `preset::production` the checks compile away.
///
/// @tparam Ax Axiomatic property tag (e.g. `law::spd`, `law::self_adjoint`).
/// @tparam Mat Concrete matrix container type.
/// @param A Input matrix to wrap.
/// @param loc Call site location for diagnostics.
/// @return `structured_mat<Mat, Ax>` carrying verified property evidence.
template <class Ax, class Mat>
[[nodiscard]] inline structured_mat<Mat, Ax>
assume_property(Mat A, std::source_location loc = std::source_location::current()) {
    if (A.rows() == A.cols()) {
        const props_detail::matrix_as_operator<Mat> view{A};
        verify_property<Ax, basic_vec<entry_t<Mat>>>(view, loc);
    }
    return structured_mat<Mat, Ax>(
        std::move(A), {math::evidence_origin::assumed, loc, "legacy sampled assertion"});
}

/// @brief Attach a square matrix guarantee without checking the entries.
///
/// Wraps matrix in `sq_mat<Mat>`, satisfying `square_matrix_like` concepts
/// required by direct solvers like `lu()`.
///
/// @tparam Mat Concrete matrix container type.
/// @param A mat whose squareness is asserted.
/// @return `sq_mat<Mat>` wrapper carrying square shape evidence.
/// @see make_square
template <class Mat = mat>
[[nodiscard]] inline sq_mat<Mat> assume_square(Mat A) {
    return sq_mat<Mat>(std::move(A));
}

/// @brief Attach a symmetry guarantee \f$A = A^T\f$, sampled under the active diagnostic preset.
///
/// Verifies symmetry via random vector probing under non-production presets.
///
/// @tparam Mat Concrete matrix container type.
/// @param A Symmetric matrix to wrap.
/// @param loc Source location for invariant failure messages.
/// @return `sym_mat<Mat>` wrapper accepted by eigensolvers and MINRES.
/// @see make_symmetric
template <class Mat = mat>
[[nodiscard]] inline sym_mat<Mat>
assume_symmetric(Mat A, std::source_location loc = std::source_location::current()) {
    return assume_property<law::self_adjoint>(std::move(A), loc);
}

/// @brief Attach a positive semi-definiteness guarantee \f$x^T A x \ge 0\f$, sampled under active preset.
///
/// @tparam Mat Concrete matrix container type.
/// @param A PSD matrix to wrap.
/// @param loc Source location for diagnostics.
/// @return `psd_matrix<Mat>` wrapper carrying PSD evidence.
template <class Mat = mat>
[[nodiscard]] inline psd_matrix<Mat>
assume_psd(Mat A, std::source_location loc = std::source_location::current()) {
    return assume_property<law::psd>(std::move(A), loc);
}

/// @brief Attach a Symmetric Positive Definite (SPD) guarantee \f$x^T A x > 0\f$.
///
/// Samples Rayleigh quotients under the active diagnostic preset. Enables direct
/// dispatch to `num::cholesky()`, `num::cg()`, and `num::pcg()`.
///
/// @tparam Mat Concrete matrix container type.
/// @param A SPD matrix to wrap.
/// @param loc Call site location for diagnostics.
/// @return `spd_mat<Mat>` wrapper carrying SPD evidence.
/// @see make_spd
template <class Mat = mat>
[[nodiscard]] inline spd_mat<Mat>
assume_spd(Mat A, std::source_location loc = std::source_location::current()) {
    return assume_property<law::spd>(std::move(A), loc);
}

/// @brief Attach a Hermitian guarantee \f$A = A^*\f$, sampled under the active diagnostic preset.
///
/// @tparam Mat Concrete matrix container type.
/// @param A Hermitian matrix to wrap.
/// @param loc Source location for diagnostics.
/// @return `hermitian_matrix<Mat>` wrapper carrying Hermitian evidence.
template <class Mat = mat>
[[nodiscard]] inline hermitian_matrix<Mat>
assume_hermitian(Mat A, std::source_location loc = std::source_location::current()) {
    return assume_property<law::self_adjoint>(std::move(A), loc);
}

/// @brief Validate square dimensions exhaustively before constructing a property wrapper.
///
/// @tparam Mat Concrete matrix container type.
/// @param A mat to validate.
/// @throws std::invalid_argument If `A.rows() != A.cols()`.
/// @return `sq_mat<Mat>` wrapper carrying verified square dimension evidence.
template <class Mat = mat>
[[nodiscard]] inline sq_mat<Mat> make_square(Mat A) {
    if (A.rows() != A.cols()) {
        throw std::invalid_argument("make_square: matrix must be square");
    }
    return sq_mat<Mat>(std::move(A));
}

/// @brief Validate symmetry \f$A = A^T\f$ exhaustively in \f$\mathcal{O}(n^2)\f$ before constructing wrapper.
///
/// @tparam Mat Concrete matrix container type.
/// @param A mat to validate.
/// @param tol Absolute entrywise symmetry tolerance.
/// @param loc Source location for diagnostics.
/// @throws std::invalid_argument If \f$\max_{i,j} |A_{ij} - A_{ji}| > \text{tol}\f$.
/// @return `sym_mat<Mat>` wrapper carrying verified symmetry evidence.
template <class Mat = mat>
[[nodiscard]] inline sym_mat<Mat>
make_symmetric(Mat A, real tol = 1e-12,
               std::source_location loc = std::source_location::current()) {
    if (!is_symmetric(A, tol)) {
        throw std::invalid_argument("make_symmetric: matrix is not symmetric");
    }
    return sym_mat<Mat>(
        std::move(A), {math::evidence_origin::verified, loc,
                       "exhaustive symmetry validator"});
}

/// @brief Validate positive definiteness exhaustively via \f$\mathcal{O}(n^3)\f$ Cholesky factorization.
///
/// Tests entrywise symmetry and executes unblocked Cholesky factorization. Fails if
/// any pivot \f$L_{ii} \le 0\f$ or NaN is encountered.
///
/// @tparam Mat Concrete matrix container type.
/// @param A mat to validate.
/// @param tol Absolute symmetry tolerance.
/// @param loc Source location for diagnostics.
/// @throws std::invalid_argument If the matrix is not symmetric positive definite.
/// @return `spd_mat<Mat>` wrapper carrying verified SPD evidence.
template <class Mat = mat>
[[nodiscard]] inline spd_mat<Mat>
make_spd(Mat A, real tol = 1e-12,
         std::source_location loc = std::source_location::current()) {
    if (!is_spd(A, tol)) {
        throw std::invalid_argument("make_spd: matrix is not symmetric positive definite");
    }
    return spd_mat<Mat>(
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

/// @brief mat carrying an asserted band structure \f$A_{ij} = 0\f$ outside \f$-k_l \leq j-i \leq k_u\f$.
template <class Mat = mat>
class band_mat_view final {
  public:
    using banded_matrix_tag = void;

    band_mat_view(Mat A, idx lower, idx upper)
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

/// @brief tridiagonal system held as its three occupied diagonals.
template <class Vec = vec>
class tri_mat final {
  public:
    using tridiagonal_matrix_tag = void;

    tri_mat(Vec sub, Vec main, Vec super)
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
class sparse_csr_mat final {
  public:
    using sparse_csr_matrix_tag = void;

    explicit sparse_csr_mat(Mat A) : A_(std::move(A)) {}

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
template <class Mat = mat>
[[nodiscard]] inline band_mat_view<Mat>
assume_banded(Mat A, idx lower, idx upper,
              std::source_location loc = std::source_location::current()) {
    band_mat_view<Mat> tagged(std::move(A), lower, upper);
    debug::verify_banded_structure(tagged, loc);
    debug::verify_band_occupancy(tagged, 0.0, loc);
    return tagged;
}

/// @brief Assert a tridiagonal structure, checking the three diagonals are consistently sized.
template <class Vec = vec>
[[nodiscard]] inline tri_mat<Vec>
assume_tridiagonal(Vec sub, Vec main, Vec super,
                   std::source_location loc = std::source_location::current()) {
    debug::verify_tridiagonal_structure(sub, main, super, loc);
    return tri_mat<Vec>(std::move(sub), std::move(main), std::move(super));
}

/// @brief Assert valid CSR storage: monotonic row offsets, in-range columns, finite values.
template <class Mat>
[[nodiscard]] inline sparse_csr_mat<Mat>
assume_sparse_csr(Mat A, std::source_location loc = std::source_location::current()) {
    debug::verify_sparse_structure(A, loc);
    return sparse_csr_mat<Mat>(std::move(A));
}

} // namespace linear

// Expose property types and assume_* / make_* taggers in top-level num:: namespace
using linear::psd_matrix;
using linear::spd_mat;
using linear::sq_mat;
using linear::structured_mat;
using linear::sym_mat;
using linear::hermitian_matrix;

using linear::assume_hermitian;
using linear::assume_property;
using linear::assume_psd;
using linear::assume_spd;
using linear::assume_square;
using linear::assume_symmetric;

using linear::assume_banded;
using linear::assume_sparse_csr;
using linear::assume_tridiagonal;
using linear::band_mat_view;
using linear::sparse_csr_mat;
using linear::tri_mat;

using linear::make_spd;
using linear::make_square;
using linear::make_symmetric;

} // namespace num
