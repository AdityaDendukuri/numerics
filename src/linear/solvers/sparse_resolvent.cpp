#include "linear/solvers/sparse_resolvent.hpp"
#include <algorithm>
#include <climits>
#include <stdexcept>
#include <utility>

#if defined(NUMERICS_HAS_UMFPACK)
#include <umfpack.h>
#endif

namespace num {

bool sparse_resolvent_available() noexcept {
#if defined(NUMERICS_HAS_UMFPACK)
    return true;
#else
    return false;
#endif
}

// The pattern of sI - A in compressed columns, -A's values, and UMFPACK's symbolic analysis.
struct sparse_resolvent::analysis {
    idx n = 0;
#if defined(NUMERICS_HAS_UMFPACK)
    array<int> ap, ai;
    array<double> minus_a;
    array<int> diagonal;
    void *symbolic = nullptr;
    ~analysis() {
        if (symbolic) {
            umfpack_zi_free_symbolic(&symbolic);
        }
    }
#endif
};

// The numeric factor of sI - A, with the values it was computed from: UMFPACK's solve reads
// them again.
struct sparse_shifted_lu::numeric {
    std::shared_ptr<const sparse_resolvent::analysis> pattern;
#if defined(NUMERICS_HAS_UMFPACK)
    array<double> ar, az;
    void *factor = nullptr;
    ~numeric() {
        if (factor) {
            umfpack_zi_free_numeric(&factor);
        }
    }
#endif
};

sparse_resolvent::sparse_resolvent(const spmat &A, sparse_resolvent_options options) {
    if (A.n_rows() != A.n_cols()) {
        throw std::invalid_argument("sparse_resolvent requires a square matrix");
    }
    if (A.n_rows() > INT_MAX || A.nnz() > INT_MAX) {
        throw std::overflow_error("sparse_resolvent int32 interface overflow");
    }
    auto pattern = std::make_shared<analysis>();
    pattern->n = A.n_rows();
#if defined(NUMERICS_HAS_UMFPACK)
    const int n = static_cast<int>(pattern->n);
    pattern->ap.assign(n + 1, 0);
    for (idx i = 0; i < pattern->n; ++i) {
        for (idx k = A.row_ptr()[i]; k < A.row_ptr()[i + 1]; ++k) {
            ++pattern->ap[A.col_idx()[k] + 1];
        }
    }
    for (int j = 0; j < n; ++j) {
        pattern->ap[j + 1] += pattern->ap[j];
    }
    pattern->ai.resize(A.nnz());
    pattern->minus_a.resize(A.nnz());
    array<int> next = pattern->ap;
    for (idx i = 0; i < pattern->n; ++i) {
        for (idx k = A.row_ptr()[i]; k < A.row_ptr()[i + 1]; ++k) {
            const int p = next[A.col_idx()[k]]++;
            pattern->ai[p] = static_cast<int>(i);
            pattern->minus_a[p] = -A.values()[k];
        }
    }
    pattern->diagonal.assign(A.n_cols(), -1);
    for (int col = 0; col < n; ++col) {
        const int begin = pattern->ap[col];
        const int end = pattern->ap[col + 1];
        array<std::pair<int, double>> entries;
        entries.reserve(static_cast<std::size_t>(end - begin));
        for (int p = begin; p < end; ++p) {
            entries.emplace_back(pattern->ai[p], pattern->minus_a[p]);
        }
        std::sort(entries.begin(), entries.end(),
                  [](const auto &lhs, const auto &rhs) { return lhs.first < rhs.first; });
        for (int offset = 0; offset < end - begin; ++offset) {
            pattern->ai[begin + offset] = entries[offset].first;
            pattern->minus_a[begin + offset] = entries[offset].second;
            if (entries[offset].first == col) {
                pattern->diagonal[col] = begin + offset;
            }
        }
    }
    for (idx j = 0; j < pattern->n; ++j) {
        if (pattern->diagonal[j] < 0) {
            throw std::invalid_argument("sparse_resolvent requires an explicit diagonal");
        }
    }
    const array<double> zero(A.nnz(), 0.0);
    double control[UMFPACK_CONTROL], info[UMFPACK_INFO];
    umfpack_zi_defaults(control);
    if (options.symmetric_pattern) {
        control[UMFPACK_STRATEGY] = UMFPACK_STRATEGY_SYMMETRIC;
    }
    if (umfpack_zi_symbolic(n, n, pattern->ap.data(), pattern->ai.data(),
                            pattern->minus_a.data(), zero.data(), &pattern->symbolic, control,
                            info) != UMFPACK_OK) {
        throw std::runtime_error("UMFPACK complex symbolic analysis failed");
    }
#else
    (void)options;
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
#if defined(NUMERICS_HAS_UMFPACK)
    auto F = std::make_unique<sparse_shifted_lu::numeric>();
    F->pattern = R.analysis_;
    const sparse_resolvent::analysis &pattern = *F->pattern;
    // sI - A, built from -A for this shift alone.
    F->ar = pattern.minus_a;
    F->az.assign(pattern.minus_a.size(), 0.0);
    for (idx j = 0; j < pattern.n; ++j) {
        F->ar[pattern.diagonal[j]] += s.real();
        F->az[pattern.diagonal[j]] = s.imag();
    }
    double control[UMFPACK_CONTROL], info[UMFPACK_INFO];
    umfpack_zi_defaults(control);
    if (umfpack_zi_numeric(pattern.ap.data(), pattern.ai.data(), F->ar.data(), F->az.data(),
                           pattern.symbolic, &F->factor, control, info) != UMFPACK_OK) {
        throw std::runtime_error("UMFPACK complex numeric factorization failed");
    }
    return sparse_shifted_lu(std::move(F));
#else
    (void)R;
    (void)s;
    throw std::runtime_error("sparse_resolvent requires SuiteSparse UMFPACK complex support");
#endif
}

void solve(const sparse_shifted_lu &F, const vec<cplx> &b, vec<cplx> &x) {
#if defined(NUMERICS_HAS_UMFPACK)
    const sparse_resolvent::analysis &pattern = *F.impl_->pattern;
    const idx n = pattern.n;
    if (b.size() != n) {
        throw std::invalid_argument("sparse_resolvent solve: dimension mismatch");
    }
    array<double> br(n), bz(n), xr(n), xz(n);
    for (idx i = 0; i < n; ++i) {
        br[i] = b[i].real(), bz[i] = b[i].imag();
    }
    if (umfpack_zi_solve(UMFPACK_A, pattern.ap.data(), pattern.ai.data(), F.impl_->ar.data(),
                         F.impl_->az.data(), xr.data(), xz.data(), br.data(), bz.data(),
                         F.impl_->factor, nullptr, nullptr) != UMFPACK_OK) {
        throw std::runtime_error("UMFPACK complex solve failed");
    }
    if (x.size() != n) {
        x = vec<cplx>(n);
    }
    for (idx i = 0; i < n; ++i) {
        x[i] = {xr[i], xz[i]};
    }
#else
    (void)F;
    (void)b;
    (void)x;
    throw std::runtime_error("sparse_resolvent requires SuiteSparse UMFPACK complex support");
#endif
}

} // namespace num
