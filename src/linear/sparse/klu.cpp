#include "linear/sparse/klu.hpp"
#include <climits>
#include <stdexcept>
#include <string>
#include <utility>

#if defined(NUMERICS_HAS_KLU)
#include <klu.h>
#endif

namespace num {

struct klu_factorization::Impl {
    idx n = 0;
#if defined(NUMERICS_HAS_KLU)
    array<int> column_ptr;
    array<int> row_index;
    array<double> values;
    mutable klu_common common{};
    klu_symbolic *symbolic = nullptr;
    klu_numeric *numeric = nullptr;

    ~Impl() {
        if (numeric) {
            klu_free_numeric(&numeric, &common);
        }
        if (symbolic) {
            klu_free_symbolic(&symbolic, &common);
        }
    }
#endif
};

bool klu_available() noexcept {
#if defined(NUMERICS_HAS_KLU)
    return true;
#else
    return false;
#endif
}

klu_factorization::klu_factorization(const spmat &matrix) : impl_(std::make_unique<Impl>()) {
#if defined(NUMERICS_HAS_KLU)
    if (matrix.n_rows() != matrix.n_cols()) {
        throw std::invalid_argument("KLU factorization requires a square matrix");
    }
    if (matrix.n_rows() > INT_MAX || matrix.nnz() > INT_MAX) {
        throw std::overflow_error("KLU int32 interface cannot represent this sparse matrix");
    }

    impl_->n = matrix.n_rows();
    const int n = static_cast<int>(impl_->n);
    // The CSR arrays of A^T are A's compressed columns, with row indices sorted.
    const spmat columns = transpose(matrix);
    impl_->column_ptr.assign(columns.row_ptr(), columns.row_ptr() + n + 1);
    impl_->row_index.assign(columns.col_idx(), columns.col_idx() + columns.nnz());
    impl_->values.assign(columns.values(), columns.values() + columns.nnz());

    klu_defaults(&impl_->common);
    impl_->symbolic =
        klu_analyze(n, impl_->column_ptr.data(), impl_->row_index.data(), &impl_->common);
    if (!impl_->symbolic) {
        throw std::runtime_error("KLU symbolic analysis failed");
    }
    impl_->numeric = klu_factor(impl_->column_ptr.data(), impl_->row_index.data(),
                                impl_->values.data(), impl_->symbolic, &impl_->common);
    if (!impl_->numeric) {
        throw std::runtime_error("KLU numeric factorization failed");
    }
#else
    (void)matrix;
    throw std::runtime_error("Numerics was built without SuiteSparse KLU support");
#endif
}

klu_factorization::~klu_factorization() = default;
klu_factorization::klu_factorization(klu_factorization &&) noexcept = default;
klu_factorization &klu_factorization::operator=(klu_factorization &&) noexcept = default;
idx klu_factorization::size() const noexcept {
    return impl_ ? impl_->n : 0;
}

void solve(const klu_factorization &factor, const vec<real> &rhs, vec<real> &solution) {
#if defined(NUMERICS_HAS_KLU)
    if (rhs.size() != factor.impl_->n) {
        throw std::invalid_argument("KLU solve dimension mismatch");
    }
    solution = rhs;
    if (!klu_solve(factor.impl_->symbolic, factor.impl_->numeric, static_cast<int>(factor.impl_->n), 1, solution.data(),
                   &factor.impl_->common)) {
        throw std::runtime_error("KLU solve failed");
    }
#else
    (void)rhs;
    (void)solution;
    throw std::runtime_error("Numerics was built without SuiteSparse KLU support");
#endif
}

void solve(const klu_factorization &factor, const mat<real> &rhs, mat<real> &solution) {
#if defined(NUMERICS_HAS_KLU)
    if (rhs.rows() != factor.impl_->n) {
        throw std::invalid_argument("KLU block solve dimension mismatch");
    }
    array<double> column_major(rhs.size());
    for (idx column = 0; column < rhs.cols(); ++column) {
        for (idx row = 0; row < rhs.rows(); ++row) {
            column_major[(column * rhs.rows()) + row] = rhs(row, column);
        }
    }
    if (!klu_solve(factor.impl_->symbolic, factor.impl_->numeric, static_cast<int>(factor.impl_->n),
                   static_cast<int>(rhs.cols()), column_major.data(), &factor.impl_->common)) {
        throw std::runtime_error("KLU block solve failed");
    }
    solution = mat<real>(rhs.rows(), rhs.cols(), 0.0);
    for (idx column = 0; column < rhs.cols(); ++column) {
        for (idx row = 0; row < rhs.rows(); ++row) {
            solution(row, column) = column_major[(column * rhs.rows()) + row];
        }
    }
#else
    (void)rhs;
    (void)solution;
    throw std::runtime_error("Numerics was built without SuiteSparse KLU support");
#endif
}

void solve_transpose(const klu_factorization &factor, const vec<real> &rhs, vec<real> &solution) {
#if defined(NUMERICS_HAS_KLU)
    if (rhs.size() != factor.impl_->n) {
        throw std::invalid_argument("KLU transpose solve dimension mismatch");
    }
    solution = rhs;
    if (!klu_tsolve(factor.impl_->symbolic, factor.impl_->numeric, static_cast<int>(factor.impl_->n), 1, solution.data(),
                    &factor.impl_->common)) {
        throw std::runtime_error("KLU transpose solve failed");
    }
#else
    (void)rhs;
    (void)solution;
    throw std::runtime_error("Numerics was built without SuiteSparse KLU support");
#endif
}

void solve_transpose(const klu_factorization &factor, const mat<real> &rhs, mat<real> &solution) {
#if defined(NUMERICS_HAS_KLU)
    if (rhs.rows() != factor.impl_->n) {
        throw std::invalid_argument("KLU transpose block solve dimension mismatch");
    }
    array<double> column_major(rhs.size());
    for (idx column = 0; column < rhs.cols(); ++column) {
        for (idx row = 0; row < rhs.rows(); ++row) {
            column_major[(column * rhs.rows()) + row] = rhs(row, column);
        }
    }
    if (!klu_tsolve(factor.impl_->symbolic, factor.impl_->numeric, static_cast<int>(factor.impl_->n),
                    static_cast<int>(rhs.cols()), column_major.data(), &factor.impl_->common)) {
        throw std::runtime_error("KLU transpose block solve failed");
    }
    solution = mat<real>(rhs.rows(), rhs.cols(), 0.0);
    for (idx column = 0; column < rhs.cols(); ++column) {
        for (idx row = 0; row < rhs.rows(); ++row) {
            solution(row, column) = column_major[(column * rhs.rows()) + row];
        }
    }
#else
    (void)rhs;
    (void)solution;
    throw std::runtime_error("Numerics was built without SuiteSparse KLU support");
#endif
}

} // namespace num
