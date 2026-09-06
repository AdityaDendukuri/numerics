#include "core/math/math.hpp"

#include <vector>

namespace num::math {

template <>
struct claims_of<std::vector<double>> {
    using type = type_list<law::inner_product_space>;
};

} // namespace num::math

static_assert(num::math::inner_product_space<std::vector<double>>);

int main() {
    const std::vector<double> x{3.0, 4.0};
    return num::math::norm(x) == 5.0 ? 0 : 1;
}
