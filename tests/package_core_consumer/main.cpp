#include "core/math/math.hpp"

#include <vector>

namespace num::math {

template <>
struct model_of<std::vector<double>> {
    using laws = type_list<law::inner_product_space>;
};

} // namespace num::math

static_assert(num::math::InnerProductSpace<std::vector<double>>);

int main() {
    const std::vector<double> x{3.0, 4.0};
    return num::math::norm(x) == 5.0 ? 0 : 1;
}
