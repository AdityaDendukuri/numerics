/// @file container/vector.cpp
/// @brief Explicit instantiation of the common vector type.
///
/// The operations moved to container/vector_ops.hpp, where backend selection is a
/// compile-time overload rather than a runtime switch. What remains is a single
/// instantiation, compiled only when NUMERICS_EXTERN_TEMPLATES is defined, so
/// translation units in this build share it instead of re-instantiating.

#include "container/vector.hpp"

namespace num {

template class BasicVector<double>;

} // namespace num
