/// @file 15_diffusion_evidence_cg.cpp
/// @brief A backward-Euler diffusion operator that goes straight into CG.
///
/// `operators::backward_euler_2d` builds \f$I - \Delta t\,L\f$ with Dirichlet boundaries. The
/// type already carries the SPD law, so CG accepts it with no `assume_spd` at the call site. The
/// right-hand side \f$\sin\pi x\sin\pi y\f$ is an eigenvector of the discrete operator, so CG
/// converges in one iteration.
#include <cmath>
#include <cstdio>
#include <numerics.hpp>

int main() {
    constexpr int points_per_side = 32;
    constexpr double timestep_diffusivity = 0.05;
    const num::grid2d grid{points_per_side, 1.0 / (points_per_side + 1)};

    const num::operators::backward_euler_2d system(grid.N, timestep_diffusivity);
    static_assert(num::claims<decltype(system), num::law::spd>);

    num::vec<num::real> rhs(grid.size(), 0.0);
    for (int i = 0; i < grid.N; ++i) {
        for (int j = 0; j < grid.N; ++j) {
            rhs[grid.flat(i, j)] = std::sin(num::pi * grid.x(i)) * std::sin(num::pi * grid.y(j));
        }
    }

    num::vec<num::real> solution(rhs.size(), 0.0);
    const auto result =
        num::cg(system, rhs, solution,
                num::cg_options{.tolerance = {.rtol = 1e-12}, .max_iterations = 4 * solution.size()});
    std::printf("CG converged %d, iterations %zu, residual %.1e\n", result.converged,
                static_cast<std::size_t>(result.iterations), result.residual);
    return result.converged ? 0 : 1;
}
