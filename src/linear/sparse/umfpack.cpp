#include "linear/sparse/umfpack.hpp"
#include <cstdint>
#include <stdexcept>

#if defined(NUMERICS_HAS_UMFPACK)
#include <umfpack.h>
#endif

namespace num {

namespace {

[[noreturn]] void unavailable() { throw std::runtime_error("Numerics was built without UMFPACK support"); }

#if defined(NUMERICS_HAS_UMFPACK)
/// CSC arrays (int64) of a square CSR matrix with entry values from `value(row, entry)`.
template <class T, class F>
void to_csc(const spmat &M, array<int64_t> &colptr, array<int64_t> &rowind, array<T> &vals, F value) {
    const idx n = M.n_rows();
    colptr.assign(n + 1, 0);
    for (idx row = 0; row < n; ++row) {
        for (idx e = M.row_ptr()[row]; e < M.row_ptr()[row + 1]; ++e) {
            ++colptr[M.col_idx()[e] + 1];
        }
    }
    for (idx c = 0; c < n; ++c) {
        colptr[c + 1] += colptr[c];
    }
    rowind.resize(M.nnz());
    vals.resize(M.nnz());
    array<int64_t> next(colptr.begin(), colptr.end() - 1);
    for (idx row = 0; row < n; ++row) {
        for (idx e = M.row_ptr()[row]; e < M.row_ptr()[row + 1]; ++e) {
            const int64_t d = next[M.col_idx()[e]]++;
            rowind[d] = static_cast<int64_t>(row);
            vals[d] = value(row, e);
        }
    }
}
#endif

} // namespace

bool umfpack_available() noexcept {
#if defined(NUMERICS_HAS_UMFPACK)
    return true;
#else
    return false;
#endif
}

// ---------------------------------------------------------------- real

struct umfpack_factorization::Impl {
    idx n = 0;
#if defined(NUMERICS_HAS_UMFPACK)
    array<int64_t> colptr, rowind;
    array<double> values;
    double control[UMFPACK_CONTROL]{};
    void *numeric = nullptr;
    mutable array<int64_t> wi;
    mutable array<double> w;
    ~Impl() {
        if (numeric) {
            umfpack_dl_free_numeric(&numeric);
        }
    }
    void solve(int sys, const vec<real> &b, vec<real> &x) const {
        if (b.size() != n) {
            throw std::invalid_argument("UMFPACK solve dimension mismatch");
        }
        if (x.size() != n) {
            x = vec<real>(n, 0.0);
        }
        if (umfpack_dl_wsolve(sys, colptr.data(), rowind.data(), values.data(), x.data(), b.data(), numeric, control,
                              nullptr, wi.data(), w.data()) != UMFPACK_OK) {
            throw std::runtime_error("UMFPACK solve failed");
        }
    }
#endif
};

umfpack_factorization::umfpack_factorization(const spmat &matrix) : impl_(std::make_unique<Impl>()) {
#if defined(NUMERICS_HAS_UMFPACK)
    if (matrix.n_rows() != matrix.n_cols()) {
        throw std::invalid_argument("UMFPACK factorization requires a square matrix");
    }
    Impl &f = *impl_;
    f.n = matrix.n_rows();
    to_csc(matrix, f.colptr, f.rowind, f.values, [&](idx, idx e) { return matrix.values()[e]; });
    f.wi.resize(f.n);
    f.w.resize(5 * f.n);
    umfpack_dl_defaults(f.control);
    f.control[UMFPACK_IRSTEP] = 0;
    const auto n = static_cast<int64_t>(f.n);
    void *symbolic = nullptr;
    if (umfpack_dl_symbolic(n, n, f.colptr.data(), f.rowind.data(), f.values.data(), &symbolic, f.control, nullptr) !=
        UMFPACK_OK) {
        throw std::runtime_error("UMFPACK symbolic analysis failed");
    }
    const int status =
        umfpack_dl_numeric(f.colptr.data(), f.rowind.data(), f.values.data(), symbolic, &f.numeric, f.control, nullptr);
    umfpack_dl_free_symbolic(&symbolic);
    if (status != UMFPACK_OK) {
        throw std::runtime_error("UMFPACK numeric factorization failed");
    }
#else
    (void)matrix;
    unavailable();
#endif
}

umfpack_factorization::~umfpack_factorization() = default;
umfpack_factorization::umfpack_factorization(umfpack_factorization &&) noexcept = default;
umfpack_factorization &umfpack_factorization::operator=(umfpack_factorization &&) noexcept = default;
idx umfpack_factorization::size() const noexcept { return impl_ ? impl_->n : 0; }

void solve(const umfpack_factorization &factor, const vec<real> &b, vec<real> &x) {
#if defined(NUMERICS_HAS_UMFPACK)
    factor.impl_->solve(UMFPACK_A, b, x);
#else
    (void)factor, (void)b, (void)x;
    unavailable();
#endif
}

void solve_transpose(const umfpack_factorization &factor, const vec<real> &b, vec<real> &x) {
#if defined(NUMERICS_HAS_UMFPACK)
    factor.impl_->solve(UMFPACK_At, b, x);
#else
    (void)factor, (void)b, (void)x;
    unavailable();
#endif
}

// ---------------------------------------------------------------- complex shifted

struct umfpack_shifted_lu::Impl {
    idx n = 0;
#if defined(NUMERICS_HAS_UMFPACK)
    array<int64_t> colptr, rowind;
    array<cplx> values;  // packed complex
    double control[UMFPACK_CONTROL]{};
    void *numeric = nullptr;
    ~Impl() {
        if (numeric) {
            umfpack_zl_free_numeric(&numeric);
        }
    }
    void solve(int sys, const vec<cplx> &b, vec<cplx> &x) const {
        if (b.size() != n) {
            throw std::invalid_argument("UMFPACK solve dimension mismatch");
        }
        if (x.size() != n) {
            x = vec<cplx>(n, cplx(0.0));
        }
        if (umfpack_zl_solve(sys, colptr.data(), rowind.data(), reinterpret_cast<const double *>(values.data()), nullptr,
                             reinterpret_cast<double *>(x.data()), nullptr, reinterpret_cast<const double *>(b.data()),
                             nullptr, numeric, control, nullptr) != UMFPACK_OK) {
            throw std::runtime_error("UMFPACK solve failed");
        }
    }
#endif
};

umfpack_shifted_lu::umfpack_shifted_lu(const spmat &A, cplx s) : impl_(std::make_unique<Impl>()) {
#if defined(NUMERICS_HAS_UMFPACK)
    if (A.n_rows() != A.n_cols()) {
        throw std::invalid_argument("UMFPACK shifted factorization requires a square matrix");
    }
    Impl &f = *impl_;
    f.n = A.n_rows();
    to_csc(A, f.colptr, f.rowind, f.values,
           [&](idx row, idx e) { return (A.col_idx()[e] == row ? s : cplx(0.0)) - A.values()[e]; });
    umfpack_zl_defaults(f.control);
    f.control[UMFPACK_IRSTEP] = 0;
    const auto n = static_cast<int64_t>(f.n);
    auto *x = reinterpret_cast<double *>(f.values.data());
    void *symbolic = nullptr;
    if (umfpack_zl_symbolic(n, n, f.colptr.data(), f.rowind.data(), x, nullptr, &symbolic, f.control, nullptr) !=
        UMFPACK_OK) {
        throw std::runtime_error("UMFPACK symbolic analysis failed");
    }
    const int status =
        umfpack_zl_numeric(f.colptr.data(), f.rowind.data(), x, nullptr, symbolic, &f.numeric, f.control, nullptr);
    umfpack_zl_free_symbolic(&symbolic);
    if (status != UMFPACK_OK) {
        throw std::runtime_error("UMFPACK numeric factorization failed (singular shift? A must store its diagonal)");
    }
#else
    (void)A, (void)s;
    unavailable();
#endif
}

umfpack_shifted_lu::~umfpack_shifted_lu() = default;
umfpack_shifted_lu::umfpack_shifted_lu(umfpack_shifted_lu &&) noexcept = default;
umfpack_shifted_lu &umfpack_shifted_lu::operator=(umfpack_shifted_lu &&) noexcept = default;
idx umfpack_shifted_lu::size() const noexcept { return impl_ ? impl_->n : 0; }

void solve(const umfpack_shifted_lu &factor, const vec<cplx> &b, vec<cplx> &x) {
#if defined(NUMERICS_HAS_UMFPACK)
    factor.impl_->solve(UMFPACK_A, b, x);
#else
    (void)factor, (void)b, (void)x;
    unavailable();
#endif
}

void solve_transpose(const umfpack_shifted_lu &factor, const vec<cplx> &b, vec<cplx> &x) {
#if defined(NUMERICS_HAS_UMFPACK)
    factor.impl_->solve(UMFPACK_Aat, b, x);
#else
    (void)factor, (void)b, (void)x;
    unavailable();
#endif
}

} // namespace num
