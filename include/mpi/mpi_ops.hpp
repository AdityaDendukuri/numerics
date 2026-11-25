/// @file mpi/mpi_ops.hpp
/// @brief Distributed operations over MPI.
#pragma once


#include "container/matrix.hpp"
#include "container/vector.hpp"
#include "container/vector_ops.hpp"

#ifdef NUMERICS_HAS_MPI
#include <mpi.h>
#else
using MPI_Comm = int;
constexpr MPI_Comm MPI_COMM_WORLD = 0;
#endif

namespace num::mpi {

/// @brief Initialize MPI; call once.
void init(int *argc, char ***argv);

/// @brief Finalize MPI.
void finalize();

/// @brief Rank in `comm`.
int rank(MPI_Comm comm = MPI_COMM_WORLD);

/// @brief Size of `comm`.
int size(MPI_Comm comm = MPI_COMM_WORLD);

/// @brief Distributed dot product: \f$\sum_r x_r^T y_r\f$ over ranks.
real dot(const vec<real> &x, const vec<real> &y, MPI_Comm comm = MPI_COMM_WORLD);

/// @brief Distributed Euclidean norm: \f$\|x\|_2\f$ over ranks.
real norm(const vec<real> &x, MPI_Comm comm = MPI_COMM_WORLD);

/// @brief Allreduce sum: \f$x \leftarrow \sum_r x_r\f$.
void allreduce_sum(real *data, idx n, MPI_Comm comm = MPI_COMM_WORLD);

/// @brief Broadcast from `root`.
void broadcast(real *data, idx n, int root = 0, MPI_Comm comm = MPI_COMM_WORLD);


#ifndef NUMERICS_HAS_MPI

// Serial fallback. Without MPI a run is a single rank holding the whole vector,
// so the collectives reduce to their local case and the point-to-point calls
// have nothing to do. Defined here rather than in a compiled stub so a build
// that never links numerics::mpi still resolves these.

inline void init(int *, char ***) {}
inline void finalize() {}

inline int rank(MPI_Comm) {
    return 0;
}

inline int size(MPI_Comm) {
    return 1;
}

inline real dot(const vec<real> &x, const vec<real> &y, MPI_Comm) {
    return num::dot(x, y);
}

inline real norm(const vec<real> &x, MPI_Comm) {
    return num::norm(x);
}

inline void allreduce_sum(real *, idx, MPI_Comm) {}
inline void broadcast(real *, idx, int, MPI_Comm) {}

#endif // NUMERICS_HAS_MPI

} // namespace num::mpi
