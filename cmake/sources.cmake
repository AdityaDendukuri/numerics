# Compiled sources for the numerics library.
#
# numerics is header-only apart from what is listed here. A file earns a place
# in this list for one of two reasons: it binds an external library whose
# headers should not reach every consumer (KLU, UMFPACK, MPI, CUDA), or it is a
# translation unit whose file-local helpers are specific enough to the algorithm
# that inlining them into a header would export detail without buying anything.
#
# NUMERICS_HAS_CUDA and NUMERICS_HAS_MPI must be set before this file is
# included (see cmake/cuda.cmake, cmake/mpi.cmake).

set(NUMERICS_CORE_SOURCES
    # Containers. Compiled so the build shares one instantiation of the common
    # element types across translation units (see NUMERICS_EXTERN_TEMPLATES).
    src/container/matrix.cpp
    src/container/vector.cpp

    # External bindings.
    src/linear/sparse/klu.cpp
    src/spectral/fft.cpp
    src/linear/sparse/umfpack.cpp

    # Algorithm-local translation units.
    src/linear/eigen/eig.cpp
    src/linear/factorization/inverse_diagonal.cpp
    src/linear/solvers/auto_linear.cpp
    src/linear/solvers/dense_resolvent.cpp
    src/linear/solvers/hessenberg_resolvent.cpp
    src/linear/solvers/resolvent.cpp
    src/linear/solvers/sparse_resolvent.cpp
    src/pde/poisson.cpp
)

# Explicit instantiations of the container templates. Kept separate so a
# consumer copying headers out of the tree is never linked against them.
set(NUMERICS_INSTANTIATION_SOURCES
)

# Distributed-memory bindings. Built as numerics::mpi, never folded into the
# main library, so linking numerics does not pull in an MPI dependency.
set(NUMERICS_MPI_SOURCES
    include/mpi/mpi_ops.cpp
)

# Device bindings. Built as numerics::cuda only when a toolkit was found.
set(NUMERICS_CUDA_SOURCES
    include/cuda/cuda_ops.cu
)
