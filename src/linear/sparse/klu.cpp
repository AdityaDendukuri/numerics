#include "linear/sparse/klu.hpp"
#include <climits>
#include <stdexcept>
#include <string>
#include <utility>

#if defined(NUMERICS_HAS_KLU)
#include <klu.h>
#endif

namespace num {

struct KLUFactor::Impl {
    idx n = 0;
#if defined(NUMERICS_HAS_KLU)
    std::vector<int> column_ptr;
    std::vector<int> row_index;
    std::vector<double> values;
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

KLUFactor::KLUFactor(const SparseMatrix &matrix) : impl_(std::make_unique<Impl>()) {
#if defined(NUMERICS_HAS_KLU)
    if (matrix.n_rows() != matrix.n_cols()) {
        throw std::invalid_argument("KLU factorization requires a square matrix");
    }
    if (matrix.n_rows() > INT_MAX || matrix.nnz() > INT_MAX) {
        throw std::overflow_error("KLU int32 interface cannot represent this sparse matrix");
    }

    impl_->n = matrix.n_rows();
    const int n = static_cast<int>(impl_->n);
    impl_->column_ptr.assign(n + 1, 0);
    for (idx row = 0; row < matrix.n_rows(); ++row) {
        for (idx entry = matrix.row_ptr()[row]; entry < matrix.row_ptr()[row + 1]; ++entry) {
            ++impl_->column_ptr[matrix.col_idx()[entry] + 1];
        }
    }
    for (int column = 0; column < n; ++column) {
        impl_->column_ptr[column + 1] += impl_->column_ptr[column];
    }

    impl_->row_index.resize(matrix.nnz());
    impl_->values.resize(matrix.nnz());
    std::vector<int> next = impl_->column_ptr;
    for (idx row = 0; row < matrix.n_rows(); ++row) {
        for (idx entry = matrix.row_ptr()[row]; entry < matrix.row_ptr()[row + 1]; ++entry) {
            const auto column = static_cast<int>(matrix.col_idx()[entry]);
            const int destination = next[column]++;
            impl_->row_index[destination] = static_cast<int>(row);
            impl_->values[destination] = matrix.values()[entry];
        }
    }

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

KLUFactor::~KLUFactor() = default;
KLUFactor::KLUFactor(KLUFactor &&) noexcept = default;
KLUFactor &KLUFactor::operator=(KLUFactor &&) noexcept = default;
idx KLUFactor::size() const noexcept {
    return impl_ ? impl_->n : 0;
}

void KLUFactor::solve(const Vector &rhs, Vector &solution) const {
#if defined(NUMERICS_HAS_KLU)
    if (rhs.size() != impl_->n) {
        throw std::invalid_argument("KLU solve dimension mismatch");
    }
    solution = rhs;
    if (!klu_solve(impl_->symbolic, impl_->numeric, static_cast<int>(impl_->n), 1, solution.data(),
                   &impl_->common)) {
        throw std::runtime_error("KLU solve failed");
    }
#else
    (void)rhs;
    (void)solution;
    throw std::runtime_error("Numerics was built without SuiteSparse KLU support");
#endif
}

void KLUFactor::solve(const Matrix &rhs, Matrix &solution) const {
#if defined(NUMERICS_HAS_KLU)
    if (rhs.rows() != impl_->n) {
        throw std::invalid_argument("KLU block solve dimension mismatch");
    }
    std::vector<double> column_major(rhs.size());
    for (idx column = 0; column < rhs.cols(); ++column) {
        for (idx row = 0; row < rhs.rows(); ++row) {
            column_major[(column * rhs.rows()) + row] = rhs(row, column);
        }
    }
    if (!klu_solve(impl_->symbolic, impl_->numeric, static_cast<int>(impl_->n),
                   static_cast<int>(rhs.cols()), column_major.data(), &impl_->common)) {
        throw std::runtime_error("KLU block solve failed");
    }
    solution = Matrix(rhs.rows(), rhs.cols(), 0.0);
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

void KLUFactor::solve_transpose(const Vector &rhs, Vector &solution) const {
#if defined(NUMERICS_HAS_KLU)
    if (rhs.size() != impl_->n) {
        throw std::invalid_argument("KLU transpose solve dimension mismatch");
    }
    solution = rhs;
    if (!klu_tsolve(impl_->symbolic, impl_->numeric, static_cast<int>(impl_->n), 1, solution.data(),
                    &impl_->common)) {
        throw std::runtime_error("KLU transpose solve failed");
    }
#else
    (void)rhs;
    (void)solution;
    throw std::runtime_error("Numerics was built without SuiteSparse KLU support");
#endif
}

void KLUFactor::solve_transpose(const Matrix &rhs, Matrix &solution) const {
#if defined(NUMERICS_HAS_KLU)
    if (rhs.rows() != impl_->n) {
        throw std::invalid_argument("KLU transpose block solve dimension mismatch");
    }
    std::vector<double> column_major(rhs.size());
    for (idx column = 0; column < rhs.cols(); ++column) {
        for (idx row = 0; row < rhs.rows(); ++row) {
            column_major[(column * rhs.rows()) + row] = rhs(row, column);
        }
    }
    if (!klu_tsolve(impl_->symbolic, impl_->numeric, static_cast<int>(impl_->n),
                    static_cast<int>(rhs.cols()), column_major.data(), &impl_->common)) {
        throw std::runtime_error("KLU transpose block solve failed");
    }
    solution = Matrix(rhs.rows(), rhs.cols(), 0.0);
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

void KLUFactor::solve_in_place(Vector &right_hand_side) const {
    Vector result(right_hand_side.size(), 0.0);
    solve(right_hand_side, result);
    right_hand_side = std::move(result);
}

void KLUFactor::solve_in_place(Matrix &right_hand_sides) const {
    Matrix result;
    solve(right_hand_sides, result);
    right_hand_sides = std::move(result);
}

} // namespace num
