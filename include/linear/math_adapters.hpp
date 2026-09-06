/// @file math_adapters.hpp
/// @brief Linear-algebra validators for the foundational evidence system.
#pragma once

#include "core/math/evidence.hpp"
#include "core/math/operations.hpp"
#include "kernel/kernel.hpp"
#include "linear/matrix_properties.hpp"
#include <stdexcept>

namespace num {

/// Dense storage participates in the map protocol without pretending that every
/// matrix value is positive definite or even square.
template <std::floating_point T>
inline void tag_invoke(math::apply_t, const basic_mat<T> &matrix, const basic_vec<T> &x,
                       basic_vec<T> &y) {
    if (x.size() != matrix.cols()) {
        throw std::invalid_argument("math::apply: dense matrix input dimension mismatch");
    }
    if (y.size() != matrix.rows()) {
        y = basic_vec<T>(matrix.rows());
    }
    kernel::matvec(y.data(), matrix.data(), x.data(), matrix.rows(), matrix.cols());
}

} // namespace num

namespace num::linear {

template <class Mat, class Ax, class X, class Y>
inline void tag_invoke(math::apply_t, const structured_mat<Mat, Ax> &matrix, const X &x, Y &y) {
    math::apply(matrix.base(), x, y);
}

} // namespace num::linear

namespace num::math {

template <std::floating_point T>
struct claims_of<basic_mat<T>> {
    using type = type_list<law::linear_map>;
};

/// A structured matrix claims exactly the law it was tagged with. Because the lattice
/// carries implication, that one entry also supplies `linear_map` and everything else
/// weaker — there is no mapping table to keep in step, and nothing is lost in
/// translation the way it was when operator laws and evidence axioms were separate
/// hierarchies (`unitary`, `skew_adjoint` and `projection` all used to collapse to
/// `linear_map` crossing that boundary).
template <class Mat, class Ax>
struct claims_of<linear::structured_mat<Mat, Ax>> {
    using type = type_list<Ax>;
};

namespace detail {

template <std::floating_point T>
struct domain_of<basic_mat<T>, void> {
    using type = basic_vec<T>;
};

template <std::floating_point T>
struct codomain_of<basic_mat<T>, void> {
    using type = basic_vec<T>;
};

template <class Mat, class Ax>
struct domain_of<linear::structured_mat<Mat, Ax>, void> : domain_of<Mat> {};

template <class Mat, class Ax>
struct codomain_of<linear::structured_mat<Mat, Ax>, void> : codomain_of<Mat> {};

} // namespace detail


template <>
struct evidence_validator<mat, law::spd> {
    static constexpr bool available = true;

    [[nodiscard]] static bool verify(const mat &matrix) { return linear::is_spd(matrix); }
};

} // namespace num::math
