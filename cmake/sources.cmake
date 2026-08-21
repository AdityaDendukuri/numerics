# NUMERICS_SOURCES -- numerics library source list
#
# NUMERICS_HAS_CUDA and NUMERICS_HAS_MPI must be set before
# this file is included (see cmake/cuda.cmake, cmake/mpi.cmake).

# Level 1 + Level 2 Kernel Sources (Data structures, operators, 0 solver overhead)
set(NUMERICS_KERNEL_SOURCES
    src/kernel/array.cpp
    src/kernel/reduce.cpp
    src/kernel/dense.cpp
    src/kernel/subspace.cpp
    src/core/vector.cpp
    src/core/matrix.cpp
    src/linalg/sparse/sparse.cpp
    src/linalg/sparse/sparse_op.cpp
    src/linalg/banded/banded.cpp
    src/operator/operator.cpp
    src/fields/fields.cpp
)

# Level 3 Core & Solver Module Sources
set(NUMERICS_CORE_SOURCES
    src/analysis/roots.cpp
    src/analysis/quadrature.cpp

    src/stats/stats.cpp

    src/linalg/factorization/cholesky.cpp
    src/linalg/factorization/inverse_diagonal.cpp
    src/linalg/factorization/lu.cpp
    src/linalg/factorization/qr.cpp
    src/linalg/factorization/thomas.cpp
    src/linalg/factorization/tridiag_complex.cpp
    src/linalg/sparse/klu.cpp
    src/linalg/sparse/umfpack.cpp

    src/linalg/eigen/power.cpp
    src/linalg/eigen/eig.cpp
    src/linalg/eigen/lanczos.cpp

    src/linalg/svd/svd.cpp

    src/linalg/solvers/cg.cpp
    src/linalg/solvers/auto_linear.cpp
    src/linalg/solvers/auto_resolvent.cpp
    src/linalg/solvers/gauss_seidel.cpp
    src/linalg/solvers/jacobi.cpp
    src/linalg/solvers/gmres.cpp
    src/linalg/expv/expv.cpp
    src/linalg/solvers/sparse_resolvent.cpp
    src/linalg/solvers/dense_resolvent.cpp
)

if(NUMERICS_HAS_CUDA)
    list(APPEND NUMERICS_CORE_SOURCES src/core/parallel/cuda_ops.cu)
else()
    list(APPEND NUMERICS_CORE_SOURCES src/core/parallel/cuda_stubs.cpp)
endif()

if(NUMERICS_HAS_MPI)
    list(APPEND NUMERICS_CORE_SOURCES src/core/parallel/mpi_ops.cpp)
else()
    list(APPEND NUMERICS_CORE_SOURCES src/core/parallel/mpi_stubs.cpp)
endif()

# Additional Domain Sources
set(NUMERICS_PDE_SOURCES
    src/pde/field_solver.cpp
    src/pde/poisson.cpp
)

set(NUMERICS_STOCHASTIC_SOURCES
)

set(NUMERICS_SPECTRAL_SOURCES
    src/spectral/fft.cpp
)

set(NUMERICS_ODE_SOURCES
    src/ode/ode.cpp
)

# Combined source list for umbrella library target (numerics::numerics)
set(NUMERICS_SOURCES
    ${NUMERICS_KERNEL_SOURCES}
    ${NUMERICS_CORE_SOURCES}
    ${NUMERICS_PDE_SOURCES}
    ${NUMERICS_STOCHASTIC_SOURCES}
    ${NUMERICS_SPECTRAL_SOURCES}
    ${NUMERICS_ODE_SOURCES}
)

set(NUMERICS_IO_SOURCES
    src/io/json.cpp
    src/io/sparse_json.cpp
)
