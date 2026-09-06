/// @file 15_diffusion_evidence_cg.cpp
/// @brief End-to-end invariant propagation from a PDE construction into CG.

#include "numerics.hpp"

#include <cmath>
#include <iostream>

int main() {
    constexpr int points_per_side = 32;
    constexpr double timestep_diffusivity = 0.05;

    const num::grid2d grid{points_per_side, 1.0 / (points_per_side + 1)};

    // The Dirichlet backward-Euler construction establishes that I - dt*L is
    // positive definite.  That evidence is part of the returned operator, so
    // CG accepts it directly: there is no assume_spd() at the call site.
    const num::operators::backward_euler_2d system(grid.N, timestep_diffusivity);
    static_assert(num::claims<decltype(system), num::law::spd>);

    num::vec rhs(grid.size(), 0.0);
    for (int i = 0; i < grid.N; ++i) {
        for (int j = 0; j < grid.N; ++j) {
            const double x = grid.x(i);
            const double y = grid.y(j);
            rhs[grid.flat(i, j)] = std::sin(num::pi * x) * std::sin(num::pi * y);
        }
    }

    num::vec solution(rhs.size(), 0.0);
    const auto result =
        num::cg(system, rhs, solution,
                num::cg_options{.tolerance = 1e-12, .max_iterations = 4 * solution.size()});

    std::cout << "backward-euler_method diffusion: " << result.iterations << " cg_method iterations, residual "
              << result.residual << '\n';
    return result.converged ? 0 : 1;
}
