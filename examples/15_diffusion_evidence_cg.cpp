/// @file 15_diffusion_evidence_cg.cpp
/// @brief End-to-end invariant propagation from a PDE construction into CG.

#include "numerics.hpp"

#include <cmath>
#include <iostream>

int main() {
    constexpr int points_per_side = 32;
    constexpr double timestep_diffusivity = 0.05;

    const num::Grid2D grid{points_per_side, 1.0 / (points_per_side + 1)};

    // The Dirichlet backward-Euler construction establishes that I - dt*L is
    // positive definite.  That evidence is part of the returned operator, so
    // CG accepts it directly: there is no assume_spd() at the call site.
    const num::operators::BackwardEuler2D system(grid.N, timestep_diffusivity);
    static_assert(num::math::Carries<decltype(system), num::axiom::positive_definite>);

    num::Vector rhs(grid.size(), 0.0);
    for (int i = 0; i < grid.N; ++i) {
        for (int j = 0; j < grid.N; ++j) {
            const double x = grid.x(i);
            const double y = grid.y(j);
            rhs[grid.flat(i, j)] = std::sin(num::pi * x) * std::sin(num::pi * y);
        }
    }

    num::Vector solution(rhs.size(), 0.0);
    const auto result =
        num::cg(system, rhs, solution,
                num::CGOptions{.tolerance = 1e-12, .max_iterations = 4 * solution.size()});

    std::cout << "backward-Euler diffusion: " << result.iterations << " CG iterations, residual "
              << result.residual << '\n';
    return result.converged ? 0 : 1;
}
