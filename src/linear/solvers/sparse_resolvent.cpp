#include "linear/solvers/sparse_resolvent.hpp"
#include <algorithm>
#include <climits>
#include <stdexcept>
#include <utility>

#if defined(NUMERICS_HAS_KLU)
#include <klu.h>
#endif

namespace num {

bool sparse_resolvent_available() noexcept {
#if defined(NUMERICS_HAS_KLU)
    return true;
#else
    return false;
#endif
}

// The pattern of sI - A in compressed columns, -A's values, and KLU's symbolic analysis. The
// analysis depends only on the pattern, so every shift reuses it; KLU's numeric factorization
// only reads it, so shifts may be factored concurrently.
struct sparse_resolvent::analysis {
    idx n = 0;
#if defined(NUMERICS_HAS_KLU)
    array<int> ap, ai;
    array<double> minus_a;
    array<int> diagonal;
    klu_symbolic *symbolic = nullptr;
    ~analysis() {
        if (symbolic) {
            klu_common common;
            klu_defaults(&common);
            klu_free_symbolic(&symbolic, &common);
        }
    }
#endif
};

// The numeric factor of sI - A. A solve writes into the factor's workspace, so one factor
// serves one thread at a time.
struct sparse_shifted_lu::numeric {
    std::shared_ptr<const sparse_resolvent::analysis> pattern;
#if defined(NUMERICS_HAS_KLU)
    klu_numeric *factor = nullptr;
    mutable klu_common common{};
    ~numeric() {
        if (factor) {
            klu_z_free_numeric(&factor, &common);
        }
    }
#endif
};

sparse_resolvent::sparse_resolvent(const spmat &A) {
    if (A.n_rows() != A.n_cols()) {
        throw std::invalid_argument("sparse_resolvent requires a square matrix");
    }
    if (A.n_rows() > INT_MAX || A.nnz() > INT_MAX) {
        throw std::overflow_error("sparse_resolvent int32 interface overflow");
    }
    auto pattern = std::make_shared<analysis>();
    pattern->n = A.n_rows();
#if defined(NUMERICS_HAS_KLU)
    const int n = static_cast<int>(pattern->n);
    const spmat columns = transpose(A);
    pattern->ap.resize(n + 1);
    pattern->ai.resize(A.nnz());
    pattern->minus_a.resize(A.nnz());
    for (idx j = 0; j <= pattern->n; ++j) {
        pattern->ap[j] = static_cast<int>(columns.row_ptr()[j]);
    }
    pattern->diagonal.assign(A.n_cols(), -1);
    for (idx col = 0; col < pattern->n; ++col) {
        for (idx p = columns.row_ptr()[col]; p < columns.row_ptr()[col + 1]; ++p) {
            const idx row = columns.col_idx()[p];
            pattern->ai[p] = static_cast<int>(row);
            pattern->minus_a[p] = -columns.values()[p];
            if (row == col) {
                pattern->diagonal[col] = static_cast<int>(p);
            }
        }
    }
    for (idx j = 0; j < pattern->n; ++j) {
        if (pattern->diagonal[j] < 0) {
            throw std::invalid_argument("sparse_resolvent requires an explicit diagonal");
        }
    }
    klu_common common;
    klu_defaults(&common);
    pattern->symbolic = klu_analyze(n, pattern->ap.data(), pattern->ai.data(), &common);
    if (!pattern->symbolic) {
        throw std::runtime_error("KLU symbolic analysis failed");
    }
#endif
    analysis_ = std::move(pattern);
}

idx sparse_resolvent::size() const noexcept {
    return analysis_->n;
}

sparse_shifted_lu::sparse_shifted_lu(std::unique_ptr<numeric> impl) : impl_(std::move(impl)) {}
sparse_shifted_lu::~sparse_shifted_lu() = default;
sparse_shifted_lu::sparse_shifted_lu(sparse_shifted_lu &&) noexcept = default;
sparse_shifted_lu &sparse_shifted_lu::operator=(sparse_shifted_lu &&) noexcept = default;

idx sparse_shifted_lu::size() const noexcept {
    return impl_ ? impl_->pattern->n : 0;
}

sparse_shifted_lu shift(const sparse_resolvent &R, cplx s) {
#if defined(NUMERICS_HAS_KLU)
    auto F = std::make_unique<sparse_shifted_lu::numeric>();
    F->pattern = R.analysis_;
    const auto &pattern = *F->pattern;
    // sI - A from -A, as interleaved (real, imaginary) pairs; std::complex has that layout.
    array<cplx> values(pattern.minus_a.size());
    for (idx p = 0; p < values.size(); ++p) {
        values[p] = cplx(pattern.minus_a[p], 0.0);
    }
    for (idx j = 0; j < pattern.n; ++j) {
        values[pattern.diagonal[j]] += s;
    }
    klu_defaults(&F->common);
    F->factor = klu_z_factor(const_cast<int *>(pattern.ap.data()), const_cast<int *>(pattern.ai.data()),
                             reinterpret_cast<double *>(values.data()), pattern.symbolic, &F->common);
    if (!F->factor) {
        throw std::runtime_error(F->common.status == KLU_SINGULAR
                                     ? "KLU complex factorization: sI - A is singular"
                                     : "KLU complex factorization failed");
    }
    return sparse_shifted_lu(std::move(F));
#else
    (void)R;
    (void)s;
    throw std::runtime_error("sparse_resolvent requires SuiteSparse KLU");
#endif
}

void solve(const sparse_shifted_lu &F, const vec<cplx> &b, vec<cplx> &x) {
#if defined(NUMERICS_HAS_KLU)
    const auto &pattern = *F.impl_->pattern;
    const idx n = pattern.n;
    if (b.size() != n) {
        throw std::invalid_argument("sparse_resolvent solve: dimension mismatch");
    }
    if (&x != &b) {
        x = b;
    }
    if (!klu_z_solve(pattern.symbolic, F.impl_->factor, static_cast<int>(n), 1,
                     reinterpret_cast<double *>(x.data()), &F.impl_->common)) {
        throw std::runtime_error("KLU complex solve failed");
    }
#else
    (void)F;
    (void)b;
    (void)x;
    throw std::runtime_error("sparse_resolvent requires SuiteSparse KLU");
#endif
}

} // namespace num
