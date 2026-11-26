# Compiled sources. numerics itself is header-only; these are the optional bindings,
# each built as its own target only when the toolkit is found.

# Distributed-memory bindings, built as numerics::mpi.
set(NUMERICS_MPI_SOURCES
    include/mpi/mpi_ops.cpp
)

# Device bindings, built as numerics::cuda.
set(NUMERICS_CUDA_SOURCES
    include/cuda/cuda_ops.cu
)
