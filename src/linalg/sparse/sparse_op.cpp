#include "linalg/sparse/sparse_op.hpp"

#include <stdexcept>

namespace num::operators {

void SparseOp::apply(const Vector &x, Vector &y) const {
    if (x.size() != A_.n_cols()) {
        throw std::invalid_argument("SparseOp::apply: input dimension mismatch");
    }
    if (y.size() != A_.n_rows()) {
        y = Vector(A_.n_rows());
    }
    sparse_matvec(A_, x, y);
}

} // namespace num::operators
