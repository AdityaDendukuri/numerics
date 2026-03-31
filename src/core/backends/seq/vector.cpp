/// @file core/backends/seq/vector.cpp
/// @brief Sequential backend -- naive serial C++ vector operations

#include "core/vector.hpp"
#include <cmath>

namespace num::backends::seq {

void scale(Vector& v, real alpha) {
    for (idx i = 0; i < v.size(); ++i)
        v[i] *= alpha;
}

void add(const Vector& x, const Vector& y, Vector& z) {
    for (idx i = 0; i < x.size(); ++i)
        z[i] = x[i] + y[i];
}

void axpy(real alpha, const Vector& x, Vector& y) {
    for (idx i = 0; i < x.size(); ++i)
        y[i] += alpha * x[i];
}

real dot(const Vector& x, const Vector& y) {
    real sum = 0;
    for (idx i = 0; i < x.size(); ++i)
        sum += x[i] * y[i];
    return sum;
}

real norm(const Vector& x) {
    real sum = 0;
    for (idx i = 0; i < x.size(); ++i)
        sum += x[i] * x[i];
    return std::sqrt(sum);
}

} // namespace num::backends::seq
