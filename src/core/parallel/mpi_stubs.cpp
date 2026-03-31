#include "core/parallel/mpi_ops.hpp"
#include <cmath>

namespace num::mpi {

void init(int*, char***) {}
void finalize() {}
int rank(MPI_Comm) {
    return 0;
}
int size(MPI_Comm) {
    return 1;
}
real dot(const Vector& x, const Vector& y, MPI_Comm) {
    return num::dot(x, y);
}
real norm(const Vector& x, MPI_Comm) {
    return num::norm(x);
}
void allreduce_sum(real*, idx, MPI_Comm) {}
void broadcast(real*, idx, int, MPI_Comm) {}

} // namespace num::mpi
