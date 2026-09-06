/// @file test_math_core_standalone.cpp
/// @brief Proves the mathematical protocol is header-only and backend-free.

#include "core/math/math.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <utility>
#include <vector>

namespace test {

struct DiagonalMap {
    using domain_type = std::vector<double>;
    using codomain_type = std::vector<double>;
    using math_laws = num::math::type_list<num::law::spd>;

    std::vector<double> diagonal;

    explicit DiagonalMap(std::vector<double> values) : diagonal(std::move(values)) {
        if (std::ranges::any_of(diagonal, [](double value) { return !(value > 0.0); })) {
            throw std::invalid_argument("DiagonalMap requires a positive diagonal");
        }
    }

    [[nodiscard]] std::size_t rows() const noexcept { return diagonal.size(); }
    [[nodiscard]] std::size_t cols() const noexcept { return diagonal.size(); }

    void apply(const std::vector<double> &x, std::vector<double> &y) const {
        for (std::size_t i = 0; i < diagonal.size(); ++i) {
            y[i] = diagonal[i] * x[i];
        }
    }
};

} // namespace test

namespace num::math {

template <>
struct claims_of<std::vector<double>> {
    using type = type_list<law::inner_product_space>;
};

template <>
struct claims_of<test::DiagonalMap> {
    using type = type_list<law::linear_map>;
};

} // namespace num::math

static_assert(num::math::inner_product_space<std::vector<double>>);
static_assert(num::math::linear_map<test::DiagonalMap>);
static_assert(num::claims<test::DiagonalMap, num::law::spd>);

int main() {
    const test::DiagonalMap map{{2.0, 3.0}};
    const std::vector<double> x{4.0, 5.0};
    auto y = num::math::zero_like(x);
    num::math::apply(map, x, y);

    num::math::linear_combination(1.0, x, -1.0, y);
    const double updated_norm_sq = num::math::axpy_norm_sq(2.0, x, y);

    const bool ok = y.size() == 2 && std::abs(y[0] - 4.0) < 1e-12 && std::abs(y[1]) < 1e-12 &&
                    std::abs(updated_norm_sq - 16.0) < 1e-12;
    if (!ok) {
        std::printf("FAIL: dependency-free mathematical apply protocol\n");
    }
    return ok ? 0 : 1;
}
