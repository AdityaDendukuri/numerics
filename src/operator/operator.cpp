#include "operator/dense.hpp"

#include <stdexcept>

namespace num::operators {

void DenseOp::apply(const Vector &x, Vector &y) const {
    if (x.size() != A_.cols()) {
        throw std::invalid_argument("DenseOp::apply: input dimension mismatch");
    }
    if (y.size() != A_.rows()) {
        y = Vector(A_.rows());
    }
    matvec(A_, x, y, b_);
}

} // namespace num::operators
