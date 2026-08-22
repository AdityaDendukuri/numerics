#include "linalg/matrix_properties.hpp"
#include "linalg/factorization/cholesky.hpp"

namespace num::linalg {

bool is_spd(const Matrix &A, real tol) {
    return is_symmetric(A, tol) && cholesky(A).success;
}

} // namespace num::linalg
