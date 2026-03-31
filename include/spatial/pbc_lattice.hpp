/// @file spatial/pbc_lattice.hpp
/// @brief Precomputed periodic-boundary neighbor arrays for a 2D square
/// lattice.
///
/// PBCLattice2D builds up[i], dn[i], lt[i], rt[i] once from modulo arithmetic
/// so the hot-path (Metropolis sweeps, BFS cluster detection) never calls %.
///
/// Flat layout: i = row * N + col,  row and col in [0, N).
#pragma once

#include <vector>

namespace num {

/// 4-neighbor periodic-boundary index arrays for an NxN lattice.
struct PBCLattice2D {
    int              N; ///< Side length; total sites = N*N
    std::vector<int> up, dn, lt,
        rt;             ///< up/dn = row +/-1, lt/rt = col +/-1 (PBC)

    explicit PBCLattice2D(int N)
        : N(N)
        , up(N * N)
        , dn(N * N)
        , lt(N * N)
        , rt(N * N) {
        for (int row = 0; row < N; ++row)
            for (int col = 0; col < N; ++col) {
                const int i = row * N + col;
                up[i]       = ((row - 1 + N) % N) * N + col;
                dn[i]       = ((row + 1) % N) * N + col;
                lt[i]       = row * N + (col - 1 + N) % N;
                rt[i]       = row * N + (col + 1) % N;
            }
    }
};

} // namespace num
