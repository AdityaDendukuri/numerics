/// @file container/matrix.cpp
/// @brief Explicit instantiation of the common matrix type.
///
/// The operations moved to container/matrix_ops.hpp, where backend selection is a
/// compile-time overload rather than a runtime switch.

#include "container/matrix.hpp"

namespace num {

template class BasicMatrix<double>;

} // namespace num
