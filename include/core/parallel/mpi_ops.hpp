/// @file mpi_ops.hpp
/// @brief MPI distributed operations
#pragma once

#include "core/matrix.hpp"
#include "core/vector.hpp"

#ifdef NUMERICS_HAS_MPI
#include <mpi.h>
#else
using MPI_Comm = int;
constexpr MPI_Comm MPI_COMM_WORLD = 0;
#endif

namespace num::mpi {

/// @brief Initialize MPI (call once)
void init(int *argc, char ***argv);

/// @brief Finalize MPI
void finalize();

/// @brief Get communicator rank
int rank(MPI_Comm comm = MPI_COMM_WORLD);

/// @brief Get communicator size
int size(MPI_Comm comm = MPI_COMM_WORLD);

/// @brief Distributed dot product (each rank holds partial vector)
real dot(const Vector &x, const Vector &y, MPI_Comm comm = MPI_COMM_WORLD);

/// @brief Distributed norm
real norm(const Vector &x, MPI_Comm comm = MPI_COMM_WORLD);

/// @brief Allreduce sum
void allreduce_sum(real *data, idx n, MPI_Comm comm = MPI_COMM_WORLD);

/// @brief Broadcast from root
void broadcast(real *data, idx n, int root = 0, MPI_Comm comm = MPI_COMM_WORLD);

} // namespace num::mpi
