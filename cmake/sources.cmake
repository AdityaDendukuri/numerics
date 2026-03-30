# LINALG_SOURCES -- numerics library source list
#
# NUMERICS_HAS_CUDA and NUMERICS_HAS_MPI must be set before
# this file is included (see cmake/cuda.cmake, cmake/mpi.cmake).
set(LINALG_SOURCES
    # Core types, constructors, and backend dispatch
    src/core/vector.cpp
    src/core/matrix.cpp

    # Core backend implementations
    src/core/backends/seq/matrix.cpp
    src/core/backends/seq/vector.cpp
    src/core/backends/opt/matrix.cpp
    src/core/backends/blas/matrix.cpp
    src/core/backends/blas/vector.cpp
    src/core/backends/omp/matrix.cpp
    src/core/backends/omp/vector.cpp
    src/core/backends/gpu/matrix.cpp
    src/core/backends/gpu/vector.cpp

    # Analysis
    src/analysis/roots.cpp
    src/analysis/quadrature.cpp

    # Statistics (simulation observables)
    src/stats/stats.cpp

    # Factorizations (dispatcher + backend impls)
    src/linalg/factorization/lu.cpp
    src/linalg/factorization/qr.cpp
    src/linalg/factorization/thomas.cpp
    src/linalg/factorization/tridiag_complex.cpp
    src/linalg/factorization/backends/seq/lu.cpp
    src/linalg/factorization/backends/seq/qr.cpp
    src/linalg/factorization/backends/seq/thomas.cpp
    src/linalg/factorization/backends/lapack/lu.cpp
    src/linalg/factorization/backends/lapack/qr.cpp
    src/linalg/factorization/backends/lapack/thomas.cpp

    # Eigenvalue solvers (dispatcher + backend impls)
    src/linalg/eigen/power.cpp
    src/linalg/eigen/eig.cpp
    src/linalg/eigen/lanczos.cpp
    src/linalg/eigen/backends/seq/jacobi_eig.cpp
    src/linalg/eigen/backends/omp/jacobi_eig.cpp
    src/linalg/eigen/backends/lapack/jacobi_eig.cpp

    # SVD (dispatcher + backend impls)
    src/linalg/svd/svd.cpp
    src/linalg/svd/backends/seq/svd.cpp
    src/linalg/svd/backends/lapack/svd.cpp

    # Iterative solvers
    src/linalg/solvers/cg.cpp
    src/linalg/solvers/gauss_seidel.cpp
    src/linalg/solvers/jacobi.cpp
    src/linalg/solvers/krylov.cpp

    # Sparse and banded
    src/linalg/sparse/sparse.cpp
    src/linalg/banded/banded.cpp

    # Spatial (cell lists, Verlet, grid)
    src/spatial/grid3d.cpp

    # PDE (stencils, fields, solvers)
    src/pde/fields.cpp

    # Markov chain Monte Carlo
    src/stochastic/backends/seq/markov.cpp
    src/stochastic/backends/omp/markov.cpp

    # Spectral transforms (FFTW3 if available, native Cooley-Tukey fallback otherwise)
    src/spectral/fft.cpp

    # ODE integrators
    src/ode/ode.cpp

    # Krylov matrix exponential (expv)
    src/linalg/expv/expv.cpp

)

# GPU backend: real implementation or CPU stub
if(NUMERICS_HAS_CUDA)
    list(APPEND LINALG_SOURCES src/core/parallel/cuda_ops.cu)
else()
    list(APPEND LINALG_SOURCES src/core/parallel/cuda_stubs.cpp)
endif()

# MPI backend: real implementation or no-op stub
if(NUMERICS_HAS_MPI)
    list(APPEND LINALG_SOURCES src/core/parallel/mpi_ops.cpp)
else()
    list(APPEND LINALG_SOURCES src/core/parallel/mpi_stubs.cpp)
endif()
