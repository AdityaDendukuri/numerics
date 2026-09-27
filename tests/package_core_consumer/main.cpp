#include "core/math/math.hpp"

#include <vector>

// A standard container is a space by its operations alone; nothing is declared.
static_assert(num::math::inner_product_space<std::vector<double>>);

int main() {
    const std::vector<double> x{3.0, 4.0};
    return num::math::norm(x) == 5.0 ? 0 : 1;
}
