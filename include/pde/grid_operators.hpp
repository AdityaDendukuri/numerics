/// @file pde/grid_operators.hpp
/// @brief mat-free grid & stencil operators with explicit spmat generation.
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

/// @brief mat-free 2D discrete 5-point Laplacian operator with spmat materialization.
class laplacian_2d final {
  public:
    using domain_type = vec;
    using codomain_type = vec;
    using math_laws = math::type_list<law::self_adjoint>;

    explicit laplacian_2d(int N) : N_(N) {
        if (N_ <= 0) {
            throw std::invalid_argument("laplacian_2d: grid dimension N must be positive");
        }
    }

    void apply(const vec &x, vec &y) const {
        const idx n = rows();
        if (x.size() != n) {
            throw std::invalid_argument("laplacian_2d: input dimension mismatch");
        }
        if (y.size() != n) {
            y = vec(n);
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

    /// @brief Materialize the 5-point discrete Laplacian as an assembled CSR spmat.
    [[nodiscard]] spmat to_sparse() const {
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
        return spmat::from_triplets(n, n, rows_idx, cols_idx, vals);
    }

  private:
    int N_;
};

/// @brief mat-free 2D Backward Euler operator \f$I - \text{coeff} \cdot \nabla^2\f$ with spmat materialization.
class backward_euler_2d final {
  public:
    using domain_type = vec;
    using codomain_type = vec;
    using math_laws = math::type_list<law::spd>;

    backward_euler_2d(int N, double coeff) : N_(N), coeff_(coeff) {
        if (N_ <= 0) {
            throw std::invalid_argument("backward_euler_2d: grid dimension N must be positive");
        }
        if (coeff_ < 0.0) {
            throw std::invalid_argument("backward_euler_2d: SPD construction requires nonnegative coefficient");
        }
    }

    void apply(const vec &x, vec &y) const {
        const idx n = rows();
        if (x.size() != n) {
            throw std::invalid_argument("backward_euler_2d: input dimension mismatch");
        }
        if (y.size() != n) {
            y = vec(n);
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

    /// @brief Materialize the Backward Euler system matrix as an assembled CSR spmat.
    [[nodiscard]] spmat to_sparse() const {
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
        return spmat::from_triplets(n, n, rows_idx, cols_idx, vals);
    }

  private:
    int N_;
    double coeff_;
};

/// @brief Generic free function to extract spmat from any sparse_convertible operator.
template <sparse_convertible Op>
[[nodiscard]] inline spmat to_sparse_matrix(const Op &op) {
    return op.to_sparse();
}

static_assert(linear_operator<laplacian_2d>);
static_assert(self_adjoint_operator<laplacian_2d>);
static_assert(sparse_convertible<laplacian_2d>);
static_assert(linear_operator<backward_euler_2d>);
static_assert(spd_operator<backward_euler_2d>);
static_assert(sparse_convertible<backward_euler_2d>);

} // namespace num::operators

namespace num::math {

template<>
struct claims_of<operators::laplacian_2d> {
    using type = type_list<law::linear_map>;
};

template<>
struct claims_of<operators::backward_euler_2d> {
    using type = type_list<law::linear_map>;
};

} // namespace num::math
