#include "core/parallel/mpi_ops.hpp"
#include <cmath>

namespace num::mpi {

void init(int* argc, char*** argv) {
    MPI_Init(argc, argv);
}
void finalize() {
    MPI_Finalize();
}

int rank(MPI_Comm comm) {
    int r;
    MPI_Comm_rank(comm, &r);
    return r;
}

int size(MPI_Comm comm) {
    int s;
    MPI_Comm_size(comm, &s);
    return s;
}

real dot(const Vector& x, const Vector& y, MPI_Comm comm) {
    real local = num::dot(x, y), global;
    MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_SUM, comm);
    return global;
}

real norm(const Vector& x, MPI_Comm comm) {
    real local_sq = num::dot(x, x), global_sq;
    MPI_Allreduce(&local_sq, &global_sq, 1, MPI_DOUBLE, MPI_SUM, comm);
    return std::sqrt(global_sq);
}

void allreduce_sum(real* data, idx n, MPI_Comm comm) {
    MPI_Allreduce(MPI_IN_PLACE,
                  data,
                  static_cast<int>(n),
                  MPI_DOUBLE,
                  MPI_SUM,
                  comm);
}

void broadcast(real* data, idx n, int root, MPI_Comm comm) {
    MPI_Bcast(data, static_cast<int>(n), MPI_DOUBLE, root, comm);
}

} // namespace num::mpi
