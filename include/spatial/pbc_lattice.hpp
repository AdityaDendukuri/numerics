/// @file spatial/pbc_lattice.hpp
/// @brief Periodic 2D square lattice with precomputed neighbours.
#pragma once

#include "core/index_space.hpp"
#include "core/types.hpp"
#include <vector>

namespace num {

/// @brief Periodic \f$N \times N\f$ lattice with neighbour arrays `up`, `dn`, `lt`, `rt`.
/// Site \f$(r, c)\f$ is \f$i = r N + c\f$.
struct pbc_lattice_2d {
    int N; ///< Side length; total sites = N*N
    array<int> up, dn, lt,
        rt; ///< up/dn = row +/-1, lt/rt = col +/-1 (PBC)

    explicit pbc_lattice_2d(int N) : N(N), up(N * N), dn(N * N), lt(N * N), rt(N * N) {
        for (int row = 0; row < N; ++row) {
            for (int col = 0; col < N; ++col) {
                const int i = (row * N) + col;
                up[i] = (((row - 1 + N) % N) * N) + col;
                dn[i] = (((row + 1) % N) * N) + col;
                lt[i] = (row * N) + ((col - 1 + N) % N);
                rt[i] = (row * N) + ((col + 1) % N);
            }
        }
    }
};

static_assert(periodic_neighbourhood_2d<pbc_lattice_2d>);
} // namespace num
