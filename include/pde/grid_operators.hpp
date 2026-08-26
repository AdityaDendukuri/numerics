/// @file pde/grid_operators.hpp
/// @brief Matrix-free grid & stencil operators with explicit SparseMatrix generation.
#pragma once

#include "algebra/properties.hpp"

#include "container/vector.hpp"
#include "core/math/evidence.hpp"
#include "core/math/models.hpp"
#include "linear/sparse/sparse.hpp"
#include "operator/concepts.hpp"
#include <stdexcept>
#include <vector>

namespace num::operators {

/// @brief Matrix-free 2D discrete 5-point Laplacian operator with SparseMatrix materialization.
class Laplacian2D final {
  public:
    using properties = property::self_adjoint;
    using domain_type = Vector;
    using codomain_type = Vector;
    using math_propositions = math::type_list<axiom::self_adjoint>;

    explicit Laplacian2D(int N) : N_(N) {
        if (N_ <= 0) {
            throw std::invalid_argument("Laplacian2D: grid dimension N must be positive");
        }
    }

    void apply(const Vector &x, Vector &y) const {
        const idx n = rows();
        if (x.size() != n) {
            throw std::invalid_argument("Laplacian2D: input dimension mismatch");
        }
        if (y.size() != n) {
            y = Vector(n);
        }
        for (int i = 0; i < N_; ++i) {
            for (int j = 0; j < N_; ++j) {
                const int k = (i * N_) + j;
                real val = -4.0 * x[k];
                if (i > 0) {
                    val += x[k - N_];
                }
                if (i < N_ - 1) {
                    val += x[k + N_];
                }
                if (j > 0) {
                    val += x[k - 1];
                }
                if (j < N_ - 1) {
                    val += x[k + 1];
                }
                y[k] = val;
            }
        }
    }

    [[nodiscard]] idx rows() const noexcept { return static_cast<idx>(N_) * N_; }
    [[nodiscard]] idx cols() const noexcept { return static_cast<idx>(N_) * N_; }
    [[nodiscard]] int N() const noexcept { return N_; }

    /// @brief Materialize the 5-point discrete Laplacian as an assembled CSR SparseMatrix.
    [[nodiscard]] SparseMatrix to_sparse() const {
        const idx n = rows();
        std::vector<idx> rows_idx, cols_idx;
        std::vector<real> vals;
        rows_idx.reserve(5 * n);
        cols_idx.reserve(5 * n);
        vals.reserve(5 * n);
        for (int i = 0; i < N_; ++i) {
            for (int j = 0; j < N_; ++j) {
                const int k = (i * N_) + j;
                rows_idx.push_back(k);
                cols_idx.push_back(k);
                vals.push_back(-4.0);
                if (i > 0) {
                    rows_idx.push_back(k);
                    cols_idx.push_back(((i - 1) * N_) + j);
                    vals.push_back(1.0);
                }
                if (i < N_ - 1) {
                    rows_idx.push_back(k);
                    cols_idx.push_back(((i + 1) * N_) + j);
                    vals.push_back(1.0);
                }
                if (j > 0) {
                    rows_idx.push_back(k);
                    cols_idx.push_back((i * N_) + (j - 1));
                    vals.push_back(1.0);
                }
                if (j < N_ - 1) {
                    rows_idx.push_back(k);
                    cols_idx.push_back((i * N_) + (j + 1));
                    vals.push_back(1.0);
                }
            }
        }
        return SparseMatrix::from_triplets(n, n, rows_idx, cols_idx, vals);
    }

  private:
    int N_;
};

/// @brief Matrix-free 2D Backward Euler operator (I - coeff * \nabla^2) with SparseMatrix materialization.
class BackwardEuler2D final {
  public:
    using properties = property::spd;
    using domain_type = Vector;
    using codomain_type = Vector;
    using math_propositions = math::type_list<axiom::positive_definite>;

    BackwardEuler2D(int N, double coeff) : N_(N), coeff_(coeff) {
        if (N_ <= 0) {
            throw std::invalid_argument("BackwardEuler2D: grid dimension N must be positive");
        }
        if (coeff_ < 0.0) {
            throw std::invalid_argument("BackwardEuler2D: SPD construction requires nonnegative coefficient");
        }
    }

    void apply(const Vector &x, Vector &y) const {
        const idx n = rows();
        if (x.size() != n) {
            throw std::invalid_argument("BackwardEuler2D: input dimension mismatch");
        }
        if (y.size() != n) {
            y = Vector(n);
        }
        const real diag = 1.0 + (4.0 * coeff_);
        for (int i = 0; i < N_; ++i) {
            for (int j = 0; j < N_; ++j) {
                const int k = (i * N_) + j;
                real val = diag * x[k];
                if (i > 0) {
                    val -= coeff_ * x[k - N_];
                }
                if (i < N_ - 1) {
                    val -= coeff_ * x[k + N_];
                }
                if (j > 0) {
                    val -= coeff_ * x[k - 1];
                }
                if (j < N_ - 1) {
                    val -= coeff_ * x[k + 1];
                }
                y[k] = val;
            }
        }
    }

    [[nodiscard]] idx rows() const noexcept { return static_cast<idx>(N_) * N_; }
    [[nodiscard]] idx cols() const noexcept { return static_cast<idx>(N_) * N_; }
    [[nodiscard]] int N() const noexcept { return N_; }
    [[nodiscard]] double coeff() const noexcept { return coeff_; }

    /// @brief Materialize the Backward Euler system matrix as an assembled CSR SparseMatrix.
    [[nodiscard]] SparseMatrix to_sparse() const {
        const idx n = rows();
        std::vector<idx> rows_idx, cols_idx;
        std::vector<real> vals;
        rows_idx.reserve(5 * n);
        cols_idx.reserve(5 * n);
        vals.reserve(5 * n);
        const real diag = 1.0 + (4.0 * coeff_);
        for (int i = 0; i < N_; ++i) {
            for (int j = 0; j < N_; ++j) {
                const int k = (i * N_) + j;
                rows_idx.push_back(k);
                cols_idx.push_back(k);
                vals.push_back(diag);
                if (i > 0) {
                    rows_idx.push_back(k);
                    cols_idx.push_back(((i - 1) * N_) + j);
                    vals.push_back(-coeff_);
                }
                if (i < N_ - 1) {
                    rows_idx.push_back(k);
                    cols_idx.push_back(((i + 1) * N_) + j);
                    vals.push_back(-coeff_);
                }
                if (j > 0) {
                    rows_idx.push_back(k);
                    cols_idx.push_back((i * N_) + (j - 1));
                    vals.push_back(-coeff_);
                }
                if (j < N_ - 1) {
                    rows_idx.push_back(k);
                    cols_idx.push_back((i * N_) + (j + 1));
                    vals.push_back(-coeff_);
                }
            }
        }
        return SparseMatrix::from_triplets(n, n, rows_idx, cols_idx, vals);
    }

  private:
    int N_;
    double coeff_;
};

/// @brief Generic free function to extract SparseMatrix from any SparseConvertible operator.
template <SparseConvertible Op>
[[nodiscard]] inline SparseMatrix to_sparse_matrix(const Op &op) {
    return op.to_sparse();
}

static_assert(LinearOperator<Laplacian2D>);
static_assert(SelfAdjointOperator<Laplacian2D>);
static_assert(SparseConvertible<Laplacian2D>);
static_assert(LinearOperator<BackwardEuler2D>);
static_assert(SPDOperator<BackwardEuler2D>);
static_assert(SparseConvertible<BackwardEuler2D>);

} // namespace num::operators

namespace num::math {

template<>
struct model_of<operators::Laplacian2D> {
    using laws = type_list<law::linear_map>;
};

template<>
struct model_of<operators::BackwardEuler2D> {
    using laws = type_list<law::linear_map>;
};

} // namespace num::math
