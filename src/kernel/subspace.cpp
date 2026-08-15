/// @file kernel/subspace.cpp
/// @brief Implementations for num::kernel::subspace.

#include "kernel/subspace.hpp"
#include <stdexcept>

namespace num::kernel::subspace {

real mgs_orthogonalize(const std::vector<Vector>& basis,
                       Vector& v,
                       std::vector<real>& h,
                       idx k) {
    for (idx i = 0; i < k; ++i) {
        h[i] = dot(v, basis[i]);
        axpy(-h[i], basis[i], v);
    }
    return norm(v);
}

real mgs_orthogonalize(const Matrix& basis, idx k, Vector& v) {
    const idx n = basis.rows();

    for (idx l = 0; l < k; ++l) {
        real proj = 0.0;
        for (idx i = 0; i < n; ++i) {
            proj += basis(i, l) * v[i];
        }
        for (idx i = 0; i < n; ++i) {
            v[i] -= proj * basis(i, l);
        }
    }
    return norm(v);
}

} // namespace num::kernel::subspace
