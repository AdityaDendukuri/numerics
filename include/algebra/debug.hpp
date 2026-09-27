/// @file algebra/debug.hpp
/// @brief Runtime sampling of the operator laws.
///
/// Sampling can only reject violations. Basis probes test a necessary condition exactly, and
/// randomized probes with a fixed seed sample away from the axes, so a failure reproduces.
#pragma once

#include "algebra/scalar.hpp"
#include "core/debug.hpp"
#include "core/types.hpp"
#include "core/math/operations.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <source_location>
#include <string>

namespace num::debug {

// ---------------------------------------------------------------------------
// Randomized probing for sampled property tests
// ---------------------------------------------------------------------------

/// @brief Number of random probe vectors drawn per sampled property test.
inline idx g_probe_count = 6;

/// @brief Relative tolerance override for sampled property tests; 0 selects sqrt(eps).
inline double g_property_tol = 0.0;

/// @brief Relative tolerance used when comparing sampled quantities over field T.
template <class T>
[[nodiscard]] inline scalars::real_t<T> property_tol() noexcept {
    return g_property_tol > 0.0 ? static_cast<scalars::real_t<T>>(g_property_tol)
                                : scalars::sampling_tol<T>();
}

/// @brief Deterministic xorshift generator, so a reported violation reproduces exactly.
struct probe_rng {
    std::uint64_t state;

    explicit constexpr probe_rng(std::uint64_t seed = 0x9E3779B97F4A7C15ULL) noexcept
        : state(seed == 0 ? 1 : seed) {}

    constexpr std::uint64_t next() noexcept {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        return state;
    }

    /// Uniform sample in [-1, 1).
    constexpr double uniform() noexcept {
        return ((static_cast<double>(next() >> 11) / 9007199254740992.0) * 2.0) - 1.0;
    }
};

/// @brief Overwrite v with a reproducible random probe over its own scalar field.
template <class VectorType>
inline void fill_probe(VectorType &v, probe_rng &rng) {
    using T = num::scalar_t<VectorType>;
    using R = scalars::real_t<T>;
    for (idx i = 0; i < v.size(); ++i) {
        if constexpr (scalars::is_complex_v<T>) {
            v[i] = T(static_cast<R>(rng.uniform()), static_cast<R>(rng.uniform()));
        } else {
            v[i] = static_cast<T>(rng.uniform());
        }
    }
}

/// @brief Overwrite v with the i-th basis vector e_i.
template <class VectorType>
inline void fill_basis(VectorType &v, idx i) {
    using T = num::scalar_t<VectorType>;
    for (idx k = 0; k < v.size(); ++k) {
        v[k] = T(0);
    }
    v[i] = T(1);
}

/// @brief Inner product \f$\langle x,y \rangle = \sum_i \overline{x_i} y_i\f$, through the type's
/// own `inner`, so a probe exercises the shipped implementation.
template <class VectorType>
[[nodiscard]] inline num::scalar_t<VectorType> probe_inner(const VectorType &x,
                                                           const VectorType &y) {
    return math::inner(x, y);
}

/// @brief Induced norm \f$\|x\| = \sqrt{\langle x,x \rangle}\f$.
template <class VectorType>
[[nodiscard]] inline scalars::real_t<num::scalar_t<VectorType>> probe_norm(const VectorType &x) {
    return std::sqrt(scalars::re(probe_inner(x, x)));
}

/// @brief Stride giving at most `cap` basis probes across dimension n.
[[nodiscard]] inline idx probe_stride(idx n, idx cap = 8) noexcept {
    const idx s = n / cap;
    return s == 0 ? idx(1) : s;
}

// ---------------------------------------------------------------------------
// Operator property sampling
// ---------------------------------------------------------------------------
//
// Generic over anything exposing `apply(x, y)` and `cols()`.

/// @brief Estimate {lambda_min, lambda_max} of a self-adjoint operator by power iteration.
///
/// The minimum comes from power-iterating \f$\lambda_{max} I - A\f$. Only this separates a
/// definite operator from a semidefinite one, since random probes never land in a null space.
template <class Op, class VectorType>
[[nodiscard]] inline auto estimate_spectrum_bounds(const Op &A, idx n, idx iterations = 64) {
    using T = num::scalar_t<VectorType>;
    using R = scalars::real_t<T>;

    VectorType v(n), w(n);
    probe_rng rng(0xD1B54A32D192ED03ULL);

    auto normalize = [&](VectorType &u) {
        const R nrm = probe_norm(u);
        if (nrm > R(0)) {
            for (idx i = 0; i < n; ++i) {
                u[i] = u[i] / static_cast<T>(nrm);
            }
        }
    };

    fill_probe(v, rng);
    normalize(v);
    R lambda_max = R(0);
    for (idx it = 0; it < iterations; ++it) {
        A.apply(v, w);
        lambda_max = scalars::re(probe_inner(v, w));
        normalize(w);
        v = w;
    }

    fill_probe(v, rng);
    normalize(v);
    R shifted = R(0);
    for (idx it = 0; it < iterations; ++it) {
        A.apply(v, w);
        for (idx i = 0; i < n; ++i) {
            w[i] = (static_cast<T>(lambda_max) * v[i]) - w[i];
        }
        shifted = scalars::re(probe_inner(v, w));
        normalize(w);
        v = w;
    }

    struct bounds {
        R min;
        R max;
    };
    return bounds{lambda_max - shifted, lambda_max};
}

/// @brief Sampled test for positive definiteness \f$\langle x, A x \rangle > 0\ \forall x \neq
/// 0\f$.
///
/// Basis probes check the necessary condition \f$A_{ii} > 0\f$ exactly; randomized
/// probes then sample the quadratic form away from the axes.
template <class Op, class VectorType>
inline void verify_spd_sample(const Op &A, idx n,
                              std::source_location loc = std::source_location::current()) {
    if constexpr (!num::debug::sampling_compiled_in) {
        return;
    }
    if (num::debug::get_level() != num::debug::diagnostic_level::full || n == 0) {
        return;
    }
    using T = num::scalar_t<VectorType>;
    using R = scalars::real_t<T>;

    VectorType x(n), Ax(n);

    const idx stride = probe_stride(n);
    for (idx i = 0; i < n; i += stride) {
        fill_basis(x, i);
        A.apply(x, Ax);
        const R diagonal = scalars::re(Ax[i]);
        if (!(diagonal > R(0))) {
            panic("PropertyError",
                  "assume_spd() assertion failed: diagonal entry A(" + std::to_string(i) + "," +
                      std::to_string(i) + ") = " + std::to_string(static_cast<double>(diagonal)) +
                      " is not positive, so the operator is NOT positive definite.",
                  loc);
        }
    }

    probe_rng rng;
    for (idx probe = 0; probe < g_probe_count; ++probe) {
        fill_probe(x, rng);
        A.apply(x, Ax);
        const R quadratic_form = scalars::re(probe_inner(x, Ax));
        if (!(quadratic_form > R(0))) {
            panic("PropertyError",
                  "assume_spd() assertion failed: sampled quadratic form Re<x,Ax> = " +
                      std::to_string(static_cast<double>(quadratic_form)) + " on probe " +
                      std::to_string(probe) +
                      " is not positive, so the operator is NOT positive definite.",
                  loc);
        }
    }

    // Definiteness is a statement about the smallest eigenvalue, which random
    // probing cannot reach: a singular positive *semi*-definite operator has a
    // null space of measure zero and passes every probe above.
    const auto bounds = estimate_spectrum_bounds<Op, VectorType>(A, n);
    const R floor_value = bounds.max * property_tol<T>();
    if (!(bounds.min > floor_value)) {
        panic("PropertyError",
              "assume_spd() assertion failed: estimated smallest eigenvalue " +
                  std::to_string(static_cast<double>(bounds.min)) + " against largest " +
                  std::to_string(static_cast<double>(bounds.max)) +
                  " indicates the operator is singular or indefinite, NOT positive definite.",
              loc);
    }
}

/// @brief Check strict diagonal dominance exactly, \f$|a_{ii}| > \sum_{j \neq i} |a_{ij}|\f$.
///
/// Reading every entry is the size of the matrix, so it is decided, not sampled. It needs
/// entrywise access; an operator exposing only `apply` is left unchecked.
template <class Mat>
inline void
verify_diagonal_dominance(const Mat &A,
                          std::source_location loc = std::source_location::current()) {
    if constexpr (!num::debug::checks_compiled_in) {
        return;
    } else if constexpr (!requires(const Mat &m) { m(idx{0}, idx{0}); }) {
        return;
    } else {
        if (num::debug::get_level() == num::debug::diagnostic_level::off) {
            return;
        }
        const idx n = A.rows();
        for (idx i = 0; i < n; ++i) {
            auto off_diagonal = scalars::mag(A(i, i));
            off_diagonal -= off_diagonal; // a zero of the right real type
            for (idx j = 0; j < n; ++j) {
                if (j != i) {
                    off_diagonal += scalars::mag(A(i, j));
                }
            }
            const auto diagonal = scalars::mag(A(i, i));
            if (!(diagonal > off_diagonal)) {
                panic("PropertyError",
                      "assume_diagonally_dominant() assertion failed: row " +
                          std::to_string(i) + " has |A(i,i)| = " +
                          std::to_string(static_cast<double>(diagonal)) +
                          ", which does not strictly exceed the off-diagonal sum " +
                          std::to_string(static_cast<double>(off_diagonal)) + ".",
                      loc);
            }
        }
    }
}

/// @brief Sampled test for positive semidefiniteness \f$\langle x, A x \rangle \geq 0\f$, which
/// admits a null space as Gram matrices and graph Laplacians need.
template <class Op, class VectorType>
inline void verify_psd_sample(const Op &A, idx n,
                              std::source_location loc = std::source_location::current()) {
    if constexpr (!num::debug::sampling_compiled_in) {
        return;
    }
    if (num::debug::get_level() != num::debug::diagnostic_level::full || n == 0) {
        return;
    }
    using T = num::scalar_t<VectorType>;
    using R = scalars::real_t<T>;

    VectorType x(n), Ax(n);
    const R tol = property_tol<T>();

    const idx stride = probe_stride(n);
    for (idx i = 0; i < n; i += stride) {
        fill_basis(x, i);
        A.apply(x, Ax);
        const R diagonal = scalars::re(Ax[i]);
        const R scale = probe_norm(Ax) + std::numeric_limits<R>::min();
        if (diagonal / scale < -tol) {
            panic("PropertyError",
                  "assume_psd() assertion failed: diagonal entry A(" + std::to_string(i) + "," +
                      std::to_string(i) + ") = " + std::to_string(static_cast<double>(diagonal)) +
                      " is negative, so the operator is NOT positive semi-definite.",
                  loc);
        }
    }

    probe_rng rng;
    for (idx probe = 0; probe < g_probe_count; ++probe) {
        fill_probe(x, rng);
        A.apply(x, Ax);
        const R quadratic_form = scalars::re(probe_inner(x, Ax));
        const R scale = (probe_norm(x) * probe_norm(Ax)) + std::numeric_limits<R>::min();
        if (quadratic_form / scale < -tol) {
            panic("PropertyError",
                  "assume_psd() assertion failed: sampled quadratic form Re<x,Ax> = " +
                      std::to_string(static_cast<double>(quadratic_form)) + " on probe " +
                      std::to_string(probe) +
                      " is negative, so the operator is NOT positive semi-definite.",
                  loc);
        }
    }
}

/// @brief Sampled test for self-adjointness \f$\langle x, A y \rangle = \overline{\langle y, A x
/// \rangle}\f$.
///
/// On a real field this is symmetry \f$A = A^T\f$; on a complex field it is the
/// Hermitian condition \f$A = A^*\f$, which conjugate-free comparison would miss.
template <class Op, class VectorType>
inline void verify_symmetry_sample(const Op &A, idx n,
                                   std::source_location loc = std::source_location::current()) {
    if constexpr (!num::debug::sampling_compiled_in) {
        return;
    }
    if (num::debug::get_level() != num::debug::diagnostic_level::full || n <= 1) {
        return;
    }
    using T = num::scalar_t<VectorType>;
    using R = scalars::real_t<T>;

    VectorType x(n), y(n), Ax(n), Ay(n);
    probe_rng rng;
    const R tol = property_tol<T>();

    for (idx probe = 0; probe < g_probe_count; ++probe) {
        fill_probe(x, rng);
        fill_probe(y, rng);
        A.apply(x, Ax);
        A.apply(y, Ay);

        const T x_A_y = probe_inner(x, Ay);
        const T y_A_x = probe_inner(y, Ax);
        const R difference = scalars::mag(x_A_y - scalars::conj(y_A_x));
        const R scale =
            std::max(scalars::mag(x_A_y), scalars::mag(y_A_x)) + std::numeric_limits<R>::min();

        if (difference / scale > tol) {
            panic("PropertyError",
                  "assume_symmetric() assertion failed: relative |<x,Ay> - conj(<y,Ax>)| = " +
                      std::to_string(static_cast<double>(difference / scale)) + " on probe " +
                      std::to_string(probe) + " exceeds tolerance " +
                      std::to_string(static_cast<double>(tol)) +
                      ", so the operator is NOT self-adjoint.",
                  loc);
        }
    }
}

/// @brief Sampled test for linearity \f$A(\alpha x + \beta y) = \alpha A x + \beta A y\f$, which every
/// law presupposes.
template <class Op, class VectorType>
inline void verify_linearity_sample(const Op &A, idx n,
                                    std::source_location loc = std::source_location::current()) {
    if constexpr (!num::debug::sampling_compiled_in) {
        return;
    }
    if (num::debug::get_level() != num::debug::diagnostic_level::full || n == 0) {
        return;
    }
    using T = num::scalar_t<VectorType>;
    using R = scalars::real_t<T>;

    VectorType x(n), y(n), combination(n), Ax(n), Ay(n), A_combination(n);
    probe_rng rng;
    const R tol = property_tol<T>();

    for (idx probe = 0; probe < g_probe_count; ++probe) {
        fill_probe(x, rng);
        fill_probe(y, rng);
        const T alpha = static_cast<T>(R(0.75));
        const T beta = static_cast<T>(R(-1.25));
        for (idx i = 0; i < n; ++i) {
            combination[i] = (alpha * x[i]) + (beta * y[i]);
        }

        A.apply(x, Ax);
        A.apply(y, Ay);
        A.apply(combination, A_combination);

        R residual_sq = R(0);
        R scale_sq = R(0);
        for (idx i = 0; i < n; ++i) {
            const T expected = (alpha * Ax[i]) + (beta * Ay[i]);
            const T d = A_combination[i] - expected;
            residual_sq += scalars::re(scalars::conj(d) * d);
            scale_sq += scalars::re(scalars::conj(expected) * expected);
        }
        const R relative =
            std::sqrt(residual_sq) / (std::sqrt(scale_sq) + std::numeric_limits<R>::min());

        if (relative > tol) {
            panic("PropertyError",
                  "Linearity check failed: relative ||A(ax+by) - (aAx+bAy)|| = " +
                      std::to_string(static_cast<double>(relative)) + " on probe " +
                      std::to_string(probe) + ", so the operator is NOT linear.",
                  loc);
        }
    }
}

} // namespace num::debug
