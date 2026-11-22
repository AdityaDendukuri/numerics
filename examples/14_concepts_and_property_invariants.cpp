/// @file 14_concepts_and_property_invariants.cpp
/// @brief Show how an SPD property is attached or checked before Cholesky.
#include <io/json.hpp>
#include <numerics.hpp>

using namespace num;

int main() {
    mat<real> A(2, 2, 0.0);
    A(0, 0) = A(1, 1) = 2.0;
    A(0, 1) = A(1, 0) = -1.0;
    const vec<real> b{1.0, 0.0};

    const auto tagged = assume_spd(A);
    static_assert(spd_operator<decltype(tagged)>);
    const vec<real> x = solve(cholesky(tagged), b);

    // make_spd checks the matrix when its property is not known in advance.
    (void)make_spd(A);

    mat<real> indefinite(2, 2, 0.0);
    indefinite(0, 0) = indefinite(1, 1) = 1.0;
    indefinite(0, 1) = indefinite(1, 0) = 3.0;
    bool rejected = false;
    try {
        (void)make_spd(indefinite);
    } catch (const std::exception &) {
        rejected = true;
    }
    io::write_json({{"solution", {x[0], x[1]}}, {"indefinite_rejected", rejected}},
                   "result.json");
}
