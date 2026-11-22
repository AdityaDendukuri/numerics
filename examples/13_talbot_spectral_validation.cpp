/// @file 13_talbot_spectral_validation.cpp
/// @brief Compare Talbot contour inversion with a Krylov solution for a birth-death process.
#include <algorithm>
#include <cmath>
#include <io/json.hpp>
#include <numerics.hpp>

using namespace num;

int main() {
    constexpr idx n = 30;
    mat<real> Q(n, n, 0.0);
    for (idx j = 0; j < n; ++j) {
        if (j + 1 < n) {
            Q(j + 1, j) = 1.0;
            Q(j, j) -= 1.0;
        }
        if (j > 0) {
            Q(j - 1, j) = 0.6;
            Q(j, j) -= 0.6;
        }
    }

    vec<real> initial(n, 0.0);
    initial[0] = 1.0;
    const real time = 1.0;
    const vec<real> reference = expv(time, operators::dense_op(Q), initial, 30, 1e-12);
    const hessenberg_resolvent reduced(Q);

    io::json results = io::json::array();
    for (idx nodes : {8, 12, 16, 24, 32}) {
        vec<cplx> result(n, 0.0);
        for (const auto &[shift, weight] : talbot_contour(time, nodes)) {
            const vec<cplx> solved = solve(num::shift(reduced, shift), initial);
            for (idx i = 0; i < n; ++i)
                result[i] += weight * solved[i];
        }

        vec<real> probability(n);
        for (idx i = 0; i < n; ++i)
            probability[i] = std::max(0.0, result[i].real());
        clip_and_normalize_nonnegative(probability);

        real error = 0.0;
        for (idx i = 0; i < n; ++i) {
            error = std::max(error, std::abs(probability[i] - reference[i]));
        }
        results.push_back({{"contour_nodes", nodes}, {"max_error", error}});
    }
    io::write_json(results, "result.json");
}
