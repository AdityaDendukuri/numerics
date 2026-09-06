/// @file operator/concepts.hpp
/// @brief Container-facing entry points to the operator half of the hierarchy.
///
/// The operator concepts — `linear_operator`, `normal_operator`, `self_adjoint_operator`,
/// `psd_operator`, `spd_operator`, `skew_adjoint_operator`, `unitary_operator`,
/// `projection_operator` — are defined once in `core/math/concepts.hpp`, where they are
/// generic over the domain and codomain and depend on nothing but the standard library.
/// They are re-exported into `num`, so `num::spd_operator` is that one definition.
///
/// This header used to define a parallel copy of that chain, parameterized on
/// `X = vec, Y = vec`. It no longer does: the hierarchy takes its domain and codomain
/// from the operator's own associated types, so `num::spd_operator<Op>` already means what
/// the copy meant, and `num::spd_operator<Op, vec, vec>` still pins them explicitly.
///
/// What is left here is what genuinely needs the container tier.
#pragma once

#include "algebra/properties.hpp"
#include "container/concepts.hpp"
#include "container/vector.hpp"
#include "core/math/concepts.hpp"
#include <type_traits>

namespace num {

/// @brief Linear operator that can materialize itself as explicit sparse CSR storage.
///
/// A statement about representation rather than mathematics: it says an operator can hand
/// over its entries, which a factorization needs and a matrix-free operator cannot do.
///
/// @tparam Op Operator type.
template <class Op>
concept sparse_convertible = linear_operator<Op> && requires(const Op &A) {
    { A.to_sparse() };
};

} // namespace num
