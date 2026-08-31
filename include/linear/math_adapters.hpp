/// @file math_adapters.hpp
/// @brief Linear-algebra validators for the foundational evidence system.
#pragma once

#include "core/math/evidence.hpp"
#include "core/math/operations.hpp"
#include "kernel/raw.hpp"
#include "linear/matrix_properties.hpp"
#include <stdexcept>

namespace num {

/// Dense storage participates in the map protocol without pretending that every
/// matrix value is positive definite or even square.
template <std::floating_point T>
inline void tag_invoke(math::apply_t, const BasicMatrix<T> &matrix, const BasicVector<T> &x,
                       BasicVector<T> &y) {
    if (x.size() != matrix.cols()) {
        throw std::invalid_argument("math::apply: dense matrix input dimension mismatch");
    }
    if (y.size() != matrix.rows()) {
        y = BasicVector<T>(matrix.rows());
    }
    kernel::raw::matvec(y.data(), matrix.data(), x.data(), matrix.rows(), matrix.cols());
}

} // namespace num

namespace num::linear {

template <class Mat, class Ax, class X, class Y>
inline void tag_invoke(math::apply_t, const StructuredMatrix<Mat, Ax> &matrix, const X &x, Y &y) {
    math::apply(matrix.base(), x, y);
}

} // namespace num::linear

namespace num::math {

template <std::floating_point T>
struct model_of<BasicMatrix<T>> {
    using laws = type_list<law::linear_map>;
};

template <class Mat, class Ax>
struct model_of<linear::StructuredMatrix<Mat, Ax>> : model_of<Mat> {};

namespace detail {

template <std::floating_point T>
struct domain_of<BasicMatrix<T>, void> {
    using type = BasicVector<T>;
};

template <std::floating_point T>
struct codomain_of<BasicMatrix<T>, void> {
    using type = BasicVector<T>;
};

template <class Mat, class Ax>
struct domain_of<linear::StructuredMatrix<Mat, Ax>, void> : domain_of<Mat> {};

template <class Mat, class Ax>
struct codomain_of<linear::StructuredMatrix<Mat, Ax>, void> : codomain_of<Mat> {};

} // namespace detail

template <class Mat, class Ax>
struct intrinsic_propositions<linear::StructuredMatrix<Mat, Ax>> {
    using type = std::conditional_t<
        std::derived_from<Ax, property::spd>, type_list<axiom::positive_definite>,
        std::conditional_t<
            std::derived_from<Ax, property::psd>, type_list<axiom::positive_semidefinite>,
            std::conditional_t<std::derived_from<Ax, property::self_adjoint>,
                               type_list<axiom::self_adjoint>, type_list<axiom::linear>>>>;
};

template <>
struct evidence_validator<Matrix, axiom::positive_definite> {
    static constexpr bool available = true;

    [[nodiscard]] static bool verify(const Matrix &matrix) { return linear::is_spd(matrix); }
};

} // namespace num::math
