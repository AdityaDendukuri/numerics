#include "linear/sparse/umfpack.hpp"
#include <climits>
#include <stdexcept>
#include <utility>

#if defined(NUMERICS_HAS_UMFPACK)
#include <umfpack.h>
#endif

namespace num {

struct umfpack_factor::Impl {
    idx n = 0;
#if defined(NUMERICS_HAS_UMFPACK)
    array<int> ap, ai;
    array<double> ax;
    void *symbolic = nullptr;
    void *numeric = nullptr;
    ~Impl() {
        if (numeric) {
            umfpack_di_free_numeric(&numeric);
        }
        if (symbolic) {
            umfpack_di_free_symbolic(&symbolic);
        }
    }
#endif
};

bool umfpack_available() noexcept {
#if defined(NUMERICS_HAS_UMFPACK)
    return true;
#else
    return false;
#endif
}

umfpack_factor::umfpack_factor(const spmat &matrix) : impl_(std::make_unique<Impl>()) {
#if defined(NUMERICS_HAS_UMFPACK)
    if (matrix.n_rows() != matrix.n_cols()) {
        throw std::invalid_argument("UMFPACK factorization requires a square matrix");
    }
    if (matrix.n_rows() > INT_MAX || matrix.nnz() > INT_MAX) {
        throw std::overflow_error("UMFPACK int32 interface cannot represent this matrix");
    }
    impl_->n = matrix.n_rows();
    const int n = static_cast<int>(impl_->n);
    impl_->ap.assign(n + 1, 0);
    for (idx row = 0; row < matrix.n_rows(); ++row) {
        for (idx k = matrix.row_ptr()[row]; k < matrix.row_ptr()[row + 1]; ++k) {
            ++impl_->ap[matrix.col_idx()[k] + 1];
        }
    }
    for (int col = 0; col < n; ++col) {
        impl_->ap[col + 1] += impl_->ap[col];
    }
    impl_->ai.resize(matrix.nnz());
    impl_->ax.resize(matrix.nnz());
    array<int> next = impl_->ap;
    for (idx row = 0; row < matrix.n_rows(); ++row) {
        for (idx k = matrix.row_ptr()[row]; k < matrix.row_ptr()[row + 1]; ++k) {
            const int col = static_cast<int>(matrix.col_idx()[k]);
            const int dest = next[col]++;
            impl_->ai[dest] = static_cast<int>(row);
            impl_->ax[dest] = matrix.values()[k];
        }
    }
    double control[UMFPACK_CONTROL], info[UMFPACK_INFO];
    umfpack_di_defaults(control);
    if (umfpack_di_symbolic(n, n, impl_->ap.data(), impl_->ai.data(), impl_->ax.data(),
                            &impl_->symbolic, control, info) != UMFPACK_OK) {
        throw std::runtime_error("UMFPACK symbolic analysis failed");
    }
    if (umfpack_di_numeric(impl_->ap.data(), impl_->ai.data(), impl_->ax.data(), impl_->symbolic,
                           &impl_->numeric, control, info) != UMFPACK_OK) {
        throw std::runtime_error("UMFPACK numeric factorization failed");
    }
#else
    (void)matrix;
    throw std::runtime_error("Numerics was built without SuiteSparse UMFPACK support");
#endif
}

umfpack_factor::~umfpack_factor() = default;
umfpack_factor::umfpack_factor(umfpack_factor &&) noexcept = default;
umfpack_factor &umfpack_factor::operator=(umfpack_factor &&) noexcept = default;
idx umfpack_factor::size() const noexcept {
    return impl_ ? impl_->n : 0;
}

void solve(const umfpack_factor &factor, const vec<real> &rhs, vec<real> &solution) {
#if defined(NUMERICS_HAS_UMFPACK)
    if (rhs.size() != factor.impl_->n) {
        throw std::invalid_argument("UMFPACK solve dimension mismatch");
    }
    // Into a fresh vector: UMFPACK reads rhs while writing x, so they cannot share storage.
    vec<real> x(factor.impl_->n, 0.0);
    const int status = umfpack_di_solve(UMFPACK_A, factor.impl_->ap.data(),
                                        factor.impl_->ai.data(), factor.impl_->ax.data(),
                                        x.data(), rhs.data(), factor.impl_->numeric, nullptr,
                                        nullptr);
    if (status != UMFPACK_OK) {
        throw std::runtime_error("UMFPACK solve failed");
    }
    solution = std::move(x);
#else
    (void)rhs;
    (void)solution;
    throw std::runtime_error("Numerics was built without SuiteSparse UMFPACK support");
#endif
}

void solve(const umfpack_factor &factor, const mat<real> &rhs, mat<real> &solution) {
#if defined(NUMERICS_HAS_UMFPACK)
    if (rhs.rows() != factor.impl_->n) {
        throw std::invalid_argument("UMFPACK block solve dimension mismatch");
    }
    mat<real> result(rhs.rows(), rhs.cols(), 0.0);
    vec<real> b(factor.impl_->n, 0.0), x;
    for (idx col = 0; col < rhs.cols(); ++col) {
        for (idx row = 0; row < rhs.rows(); ++row) {
            b[row] = rhs(row, col);
        }
        solve(factor, b, x);
        for (idx row = 0; row < rhs.rows(); ++row) {
            result(row, col) = x[row];
        }
    }
    solution = std::move(result);
#else
    (void)rhs;
    (void)solution;
    throw std::runtime_error("Numerics was built without SuiteSparse UMFPACK support");
#endif
}

} // namespace num
