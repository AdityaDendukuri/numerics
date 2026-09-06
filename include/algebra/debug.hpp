/// @file algebra/debug.hpp
/// @brief Runtime sampling of algebraic laws and operator properties.
///
/// A property quantified over all vectors cannot be proved by evaluating it a
/// finite number of times. These routines are built to reject violations. Basis
/// probes test a necessary condition exactly, and randomized probes sample the
/// property away from the coordinate axes with a fixed seed, so a reported
/// violation reproduces.
#pragma once

#include "algebra/ops.hpp"
#include "algebra/scalar.hpp"
#include "core/debug.hpp"
#include "core/types.hpp"
#include "kernel/kernel.hpp"
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
//
// A property such as symmetry or positive definiteness is a statement about
// *all* vectors. It cannot be established by evaluating a single fixed probe,
// which is why each sampler below combines deterministic basis probes (testing
// a necessary condition exactly) with several randomized probes.

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

/// @brief Hermitian inner product \f$\langle x,y \rangle = \sum_i \overline{x_i} y_i\f$.
template <class VectorType>
[[nodiscard]] inline num::scalar_t<VectorType> probe_inner(const VectorType &x,
                                                           const VectorType &y) {
    using T = num::scalar_t<VectorType>;
    if constexpr (algebra::detail::raw_reducible<VectorType>) {
        return kernel::dot(x.data(), y.data(), x.size());
    } else {
        T sum = T(0);
        for (idx i = 0; i < x.size(); ++i) {
            sum += scalars::conj(x[i]) * y[i];
        }
        return sum;
    }
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
// Algebraic law sampling
// ---------------------------------------------------------------------------
//
// The concepts in container/algebra.hpp establish that a type *supplies* the
// vector space operations and is closed under them. That it obeys the axioms --
// associativity, distributivity, conjugate symmetry, the triangle inequality --
// is not decidable from a type and is sampled here.
//
// These probes route through num::algebra, which resolves to the type's own
// operations. Checking num::vec therefore checks the shipped `dot`, `norm`,
// `axpy` and `scale`, backend dispatch included: a BLAS path that returns an
// unconjugated inner product for complex data fails conjugate symmetry here.

/// @brief Sample the abelian group axioms of vector addition.
///
/// Associativity, commutativity, the additive identity, and additive inverses.
template <class V>
inline void
verify_additive_group_axioms(idx n, std::source_location loc = std::source_location::current()) {
    if constexpr (!num::debug::sampling_compiled_in) {
        return;
    }
    if (num::debug::get_level() != num::debug::diagnostic_level::full || n == 0) {
        return;
    }
    using T = num::scalar_t<V>;
    using R = scalars::real_t<T>;
    const R tol = property_tol<T>();
    probe_rng rng;

    auto rel_diff = [&](const V &a, const V &b) {
        R num_ = R(0), den = R(0);
        for (idx i = 0; i < n; ++i) {
            const T d = a[i] - b[i];
            num_ += scalars::re(scalars::conj(d) * d);
            den += scalars::re(scalars::conj(a[i]) * a[i]);
        }
        return std::sqrt(num_) / (std::sqrt(den) + std::numeric_limits<R>::min());
    };

    for (idx probe = 0; probe < g_probe_count; ++probe) {
        V x(n), y(n), z(n);
        fill_probe(x, rng);
        fill_probe(y, rng);
        fill_probe(z, rng);

        // (x + y) + z == x + (y + z)
        V left = x;
        algebra::axpy_into(T(1), y, left);
        algebra::axpy_into(T(1), z, left);
        V right = y;
        algebra::axpy_into(T(1), z, right);
        V right_total = x;
        algebra::axpy_into(T(1), right, right_total);
        if (rel_diff(left, right_total) > tol) {
            panic("AlgebraError",
                  "vector addition is not associative: (x+y)+z != x+(y+z) on probe " +
                      std::to_string(probe),
                  loc);
        }

        // x + y == y + x
        V xy = x;
        algebra::axpy_into(T(1), y, xy);
        V yx = y;
        algebra::axpy_into(T(1), x, yx);
        if (rel_diff(xy, yx) > tol) {
            panic("AlgebraError",
                  "vector addition is not commutative: x+y != y+x on probe " +
                      std::to_string(probe),
                  loc);
        }

        // x + 0 == x
        V zero_v = algebra::zero<V>(n);
        V x_plus_zero = x;
        algebra::axpy_into(T(1), zero_v, x_plus_zero);
        if (rel_diff(x_plus_zero, x) > tol) {
            panic("AlgebraError", "the zero vector is not an additive identity: x+0 != x", loc);
        }

        // x + (-x) == 0
        V cancel = x;
        algebra::axpy_into(T(-1), x, cancel);
        R residual = R(0);
        for (idx i = 0; i < n; ++i) {
            residual += scalars::re(scalars::conj(cancel[i]) * cancel[i]);
        }
        if (std::sqrt(residual) > tol) {
            panic("AlgebraError", "additive inverses do not cancel: x + (-x) != 0", loc);
        }
    }
}

/// @brief Sample the vector space axioms relating scalar multiplication to addition.
///
/// Distributivity over vector and over scalar addition, compatibility of scalar
/// multiplication with field multiplication, and the unit scalar acting as identity.
template <class V>
inline void verify_vector_space_axioms(idx n,
                                       std::source_location loc = std::source_location::current()) {
    if constexpr (!num::debug::sampling_compiled_in) {
        return;
    }
    if (num::debug::get_level() != num::debug::diagnostic_level::full || n == 0) {
        return;
    }
    using T = num::scalar_t<V>;
    using R = scalars::real_t<T>;
    const R tol = property_tol<T>();
    probe_rng rng(0xC0FFEE123456789ULL);

    auto rel_diff = [&](const V &a, const V &b) {
        R num_ = R(0), den = R(0);
        for (idx i = 0; i < n; ++i) {
            const T d = a[i] - b[i];
            num_ += scalars::re(scalars::conj(d) * d);
            den += scalars::re(scalars::conj(a[i]) * a[i]);
        }
        return std::sqrt(num_) / (std::sqrt(den) + std::numeric_limits<R>::min());
    };

    verify_additive_group_axioms<V>(n, loc);

    for (idx probe = 0; probe < g_probe_count; ++probe) {
        V x(n), y(n);
        fill_probe(x, rng);
        fill_probe(y, rng);
        const T a = static_cast<T>(R(0.7));
        const T b = static_cast<T>(R(-1.3));

        // a(x + y) == ax + ay
        V sum = x;
        algebra::axpy_into(T(1), y, sum);
        algebra::scale_inplace(sum, a);
        V ax = x;
        algebra::scale_inplace(ax, a);
        V ay = y;
        algebra::scale_inplace(ay, a);
        V ax_plus_ay = ax;
        algebra::axpy_into(T(1), ay, ax_plus_ay);
        if (rel_diff(sum, ax_plus_ay) > tol) {
            panic("AlgebraError",
                  "scalar multiplication does not distribute over vector addition: "
                  "a(x+y) != ax+ay on probe " +
                      std::to_string(probe),
                  loc);
        }

        // (a + b)x == ax + bx
        V ab_x = x;
        algebra::scale_inplace(ab_x, a + b);
        V bx = x;
        algebra::scale_inplace(bx, b);
        V ax2 = x;
        algebra::scale_inplace(ax2, a);
        V ax_plus_bx = ax2;
        algebra::axpy_into(T(1), bx, ax_plus_bx);
        if (rel_diff(ab_x, ax_plus_bx) > tol) {
            panic("AlgebraError",
                  "scalar multiplication does not distribute over scalar addition: "
                  "(a+b)x != ax+bx on probe " +
                      std::to_string(probe),
                  loc);
        }

        // a(bx) == (ab)x
        V abx = x;
        algebra::scale_inplace(abx, b);
        algebra::scale_inplace(abx, a);
        V ab_together = x;
        algebra::scale_inplace(ab_together, a * b);
        if (rel_diff(abx, ab_together) > tol) {
            panic("AlgebraError",
                  "scalar action is not compatible with field multiplication: "
                  "a(bx) != (ab)x",
                  loc);
        }

        // 1x == x
        V one_x = x;
        algebra::scale_inplace(one_x, T(1));
        if (rel_diff(one_x, x) > tol) {
            panic("AlgebraError", "the unit scalar is not an identity: 1x != x", loc);
        }
    }
}

/// @brief Sample the inner product axioms.
///
/// Conjugate symmetry \f$\langle x,y \rangle = \overline{\langle y,x \rangle}\f$,
/// linearity in the second argument, and positive definiteness
/// \f$\langle x,x \rangle > 0\f$ for \f$x \neq 0\f$. Conjugate symmetry is the one
/// that separates a genuine Hermitian form from a transpose-only implementation,
/// and it is invisible on real data.
template <class V>
inline void
verify_inner_product_axioms(idx n, std::source_location loc = std::source_location::current()) {
    if constexpr (!num::debug::sampling_compiled_in) {
        return;
    }
    if (num::debug::get_level() != num::debug::diagnostic_level::full || n == 0) {
        return;
    }
    using T = num::scalar_t<V>;
    using R = scalars::real_t<T>;
    const R tol = property_tol<T>();
    probe_rng rng(0xABCDEF0123456789ULL);

    for (idx probe = 0; probe < g_probe_count; ++probe) {
        V x(n), y(n), z(n);
        fill_probe(x, rng);
        fill_probe(y, rng);
        fill_probe(z, rng);

        const T xy = algebra::inner(x, y);
        const T yx = algebra::inner(y, x);
        const R sym_err = scalars::mag(xy - scalars::conj(yx));
        const R sym_scale = scalars::mag(xy) + std::numeric_limits<R>::min();
        if (sym_err / sym_scale > tol) {
            panic("AlgebraError",
                  "inner product is not conjugate symmetric: <x,y> != conj(<y,x>) "
                  "(relative error " +
                      std::to_string(static_cast<double>(sym_err / sym_scale)) + ") on probe " +
                      std::to_string(probe),
                  loc);
        }

        // Linearity in the second argument: <x, ay + z> == a<x,y> + <x,z>
        const T a = static_cast<T>(R(0.6));
        V combo = z;
        algebra::axpy_into(a, y, combo);
        const T lhs = algebra::inner(x, combo);
        const T rhs = (a * algebra::inner(x, y)) + algebra::inner(x, z);
        const R lin_err = scalars::mag(lhs - rhs);
        const R lin_scale = scalars::mag(lhs) + std::numeric_limits<R>::min();
        if (lin_err / lin_scale > tol) {
            panic("AlgebraError",
                  "inner product is not linear in its second argument: "
                  "<x,ay+z> != a<x,y>+<x,z> on probe " +
                      std::to_string(probe),
                  loc);
        }

        // Positive definiteness on a nonzero vector.
        const R xx = scalars::re(algebra::inner(x, x));
        if (!(xx > R(0))) {
            panic("AlgebraError",
                  "inner product is not positive definite: <x,x> = " +
                      std::to_string(static_cast<double>(xx)) + " for a nonzero x",
                  loc);
        }
    }
}

/// @brief Sample the norm axioms, and that the norm is the one the inner product induces.
///
/// Absolute homogeneity \f$\|ax\| = |a|\,\|x\|\f$, the triangle inequality, and
/// \f$\|x\|^2 = \langle x,x \rangle\f$ -- the compatibility every Krylov method
/// assumes when it uses a residual norm to reason about an inner product.
template <class V>
inline void verify_norm_axioms(idx n, std::source_location loc = std::source_location::current()) {
    if constexpr (!num::debug::sampling_compiled_in) {
        return;
    }
    if (num::debug::get_level() != num::debug::diagnostic_level::full || n == 0) {
        return;
    }
    using T = num::scalar_t<V>;
    using R = scalars::real_t<T>;
    const R tol = property_tol<T>();
    probe_rng rng(0x13579BDF02468ACEULL);

    for (idx probe = 0; probe < g_probe_count; ++probe) {
        V x(n), y(n);
        fill_probe(x, rng);
        fill_probe(y, rng);

        const R nx = algebra::norm_of(x);
        const R ny = algebra::norm_of(y);

        // ||a x|| == |a| ||x||
        const T a = static_cast<T>(R(-2.5));
        V ax = x;
        algebra::scale_inplace(ax, a);
        const R nax = algebra::norm_of(ax);
        const R expect = scalars::mag(a) * nx;
        if (scalars::mag(nax - expect) / (expect + std::numeric_limits<R>::min()) > tol) {
            panic("AlgebraError",
                  "norm is not absolutely homogeneous: ||ax|| != |a| ||x|| on probe " +
                      std::to_string(probe),
                  loc);
        }

        // ||x + y|| <= ||x|| + ||y||
        V sum = x;
        algebra::axpy_into(T(1), y, sum);
        const R nsum = algebra::norm_of(sum);
        if (nsum > (nx + ny) * (R(1) + tol)) {
            panic("AlgebraError",
                  "norm violates the triangle inequality: ||x+y|| = " +
                      std::to_string(static_cast<double>(nsum)) +
                      " > ||x||+||y|| = " + std::to_string(static_cast<double>(nx + ny)) +
                      " on probe " + std::to_string(probe),
                  loc);
        }

        // ||x||^2 == <x,x>
        const R induced = std::sqrt(scalars::re(algebra::inner(x, x)));
        if (scalars::mag(nx - induced) / (induced + std::numeric_limits<R>::min()) > tol) {
            panic("AlgebraError",
                  "norm is not induced by the inner product: ||x|| = " +
                      std::to_string(static_cast<double>(nx)) +
                      " but sqrt(<x,x>) = " + std::to_string(static_cast<double>(induced)),
                  loc);
        }
    }
}

/// @brief Sample every law of a Hilbert space: group, scalar action, inner product, norm.
template <class V>
inline void
verify_hilbert_space_axioms(idx n, std::source_location loc = std::source_location::current()) {
    verify_vector_space_axioms<V>(n, loc);
    verify_inner_product_axioms<V>(n, loc);
    verify_norm_axioms<V>(n, loc);
}

// ---------------------------------------------------------------------------
// Operator property sampling
// ---------------------------------------------------------------------------
//
// Generic over anything exposing apply(x, y) and cols(); nothing here knows what
// an "operator" is beyond that. They live in core because the property hierarchy binds
// to them, and the hierarchy is vocabulary the whole library speaks.

/// @brief Estimate the extreme eigenvalues of a self-adjoint operator by power iteration.
///
/// Returns {lambda_min, lambda_max}. The maximum is found directly; the minimum
/// comes from power-iterating the shifted operator \f$\lambda_{max} I - A\f$, whose
/// dominant eigenvalue is \f$\lambda_{max} - \lambda_{min}\f$. This is what separates a
/// definite operator from a merely semi-definite one: a null space has measure
/// zero, so no amount of random probing will land on it.
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

/// @brief Sampled test for positive semi-definiteness \f$\langle x, A x \rangle \geq 0\f$.
///
/// Identical to the definite test but admitting a null space, as required by
/// Gram matrices and graph Laplacians.
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

/// @brief Sampled test for adjoint consistency \f$\langle A x, y \rangle = \langle x, A^* y
/// \rangle\f$.
template <class Op, class VectorType>
inline void verify_adjoint_sample(const Op &A, idx m, idx n,
                                  std::source_location loc = std::source_location::current()) {
    if constexpr (!num::debug::sampling_compiled_in) {
        return;
    }
    if (num::debug::get_level() != num::debug::diagnostic_level::full || m == 0 || n == 0) {
        return;
    }
    using T = num::scalar_t<VectorType>;
    using R = scalars::real_t<T>;

    VectorType x(n), y(m), Ax(m), Aty(n);
    probe_rng rng;
    const R tol = property_tol<T>();

    for (idx probe = 0; probe < g_probe_count; ++probe) {
        fill_probe(x, rng);
        fill_probe(y, rng);
        A.apply(x, Ax);
        A.apply_adjoint(y, Aty);

        const T lhs = probe_inner(Ax, y);
        const T rhs = probe_inner(x, Aty);
        const R difference = scalars::mag(lhs - rhs);
        const R scale =
            std::max(scalars::mag(lhs), scalars::mag(rhs)) + std::numeric_limits<R>::min();

        if (difference / scale > tol) {
            panic("PropertyError",
                  "Adjoint consistency check failed: relative |<Ax,y> - <x,A*y>| = " +
                      std::to_string(static_cast<double>(difference / scale)) + " on probe " +
                      std::to_string(probe) + " exceeds tolerance " +
                      std::to_string(static_cast<double>(tol)) + ".",
                  loc);
        }
    }
}

/// @brief Sampled test for isometry \f$\|A x\| = \|x\|\f$ (equivalently \f$A^* A = I\f$).
///
/// Basis probes check the necessary condition \f$\|A e_i\| = 1\f$ exactly, which a
/// single all-ones probe cannot detect: any A whose column norms happen to average
/// correctly would otherwise pass.
template <class Op, class VectorType>
inline void verify_orthogonal_sample(const Op &A, idx n,
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
        const R column_norm = probe_norm(Ax);
        if (scalars::mag(column_norm - R(1)) > tol) {
            panic("PropertyError",
                  "assume_orthogonal() assertion failed: column " + std::to_string(i) +
                      " has norm " + std::to_string(static_cast<double>(column_norm)) +
                      " rather than 1, so the operator is NOT an isometry.",
                  loc);
        }
    }

    probe_rng rng;
    for (idx probe = 0; probe < g_probe_count; ++probe) {
        fill_probe(x, rng);
        A.apply(x, Ax);
        const R norm_x = probe_norm(x);
        const R norm_Ax = probe_norm(Ax);
        const R scale = norm_x + std::numeric_limits<R>::min();
        if (scalars::mag(norm_Ax - norm_x) / scale > tol) {
            panic("PropertyError",
                  "assume_orthogonal() assertion failed: relative | ||Ax|| - ||x|| | = " +
                      std::to_string(static_cast<double>(scalars::mag(norm_Ax - norm_x) / scale)) +
                      " on probe " + std::to_string(probe) +
                      ", so the operator is NOT an isometry.",
                  loc);
        }
    }
}

/// @brief Sampled test for idempotency \f$P^2 = P\f$.
template <class Op, class VectorType>
inline void verify_projection_sample(const Op &P, idx n,
                                     std::source_location loc = std::source_location::current()) {
    if constexpr (!num::debug::sampling_compiled_in) {
        return;
    }
    if (num::debug::get_level() != num::debug::diagnostic_level::full || n == 0) {
        return;
    }
    using T = num::scalar_t<VectorType>;
    using R = scalars::real_t<T>;

    VectorType x(n), Px(n), PPx(n);
    probe_rng rng;
    const R tol = property_tol<T>();

    for (idx probe = 0; probe < g_probe_count; ++probe) {
        fill_probe(x, rng);
        P.apply(x, Px);
        P.apply(Px, PPx);

        R residual_sq = R(0);
        for (idx i = 0; i < n; ++i) {
            const T d = PPx[i] - Px[i];
            residual_sq += scalars::re(scalars::conj(d) * d);
        }
        const R residual = std::sqrt(residual_sq);
        const R scale = probe_norm(Px) + std::numeric_limits<R>::min();

        if (residual / scale > tol) {
            panic("PropertyError",
                  "assume_projection() assertion failed: relative ||P(Px) - Px|| = " +
                      std::to_string(static_cast<double>(residual / scale)) + " on probe " +
                      std::to_string(probe) + ", so the operator is NOT idempotent.",
                  loc);
        }
    }
}

/// @brief Sampled test for skew-adjointness \f$A = -A^*\f$, i.e. \f$\mathrm{Re}\langle x, A x
/// \rangle = 0\ \forall x\f$.
///
/// The comparison is relative to \f$\|x\|\,\|Ax\|\f$ (the Cauchy-Schwarz bound on
/// the quantity being tested), so the test does not weaken as the operator scale
/// grows. Basis probes additionally pin every sampled diagonal entry to zero,
/// which is what distinguishes a skew operator from a symmetric one that merely
/// annihilates a particular probe vector.
template <class Op, class VectorType>
inline void
verify_skew_symmetry_sample(const Op &A, idx n,
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
        const R diagonal = scalars::mag(scalars::re(Ax[i]));
        const R scale = probe_norm(Ax) + std::numeric_limits<R>::min();
        if (diagonal / scale > tol) {
            panic("PropertyError",
                  "assume_skew_symmetric() assertion failed: diagonal entry A(" +
                      std::to_string(i) + "," + std::to_string(i) +
                      ") = " + std::to_string(static_cast<double>(scalars::re(Ax[i]))) +
                      " is nonzero, so the operator is NOT skew-adjoint.",
                  loc);
        }
    }

    probe_rng rng;
    for (idx probe = 0; probe < g_probe_count; ++probe) {
        fill_probe(x, rng);
        A.apply(x, Ax);
        const R quadratic_form = scalars::mag(scalars::re(probe_inner(x, Ax)));
        const R scale = (probe_norm(x) * probe_norm(Ax)) + std::numeric_limits<R>::min();
        if (quadratic_form / scale > tol) {
            panic("PropertyError",
                  "assume_skew_symmetric() assertion failed: relative Re<x,Ax> = " +
                      std::to_string(static_cast<double>(quadratic_form / scale)) + " on probe " +
                      std::to_string(probe) + " is nonzero, so the operator is NOT skew-adjoint.",
                  loc);
        }
    }
}

/// @brief Sampled test for linearity \f$A(\alpha x + \beta y) = \alpha A x + \beta A y\f$.
///
/// Every other operator property presupposes this one, yet a stateful or
/// accidentally affine `apply` satisfies none of the tests above by contradiction.
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
