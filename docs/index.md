# numerics

`numerics` is a C++20 library of numerical methods. It has dense, banded and sparse linear
algebra, Krylov solvers, eigenvalues and the SVD, ODE integrators, PDEs, FFTs, quadrature, root
finding, graphs and sampling. The low-level loops are in `num::kernel`, which works on raw
pointers and needs nothing else. BLAS, LAPACK, OpenMP, CUDA, FFTW and SuiteSparse KLU are used if the
build finds them.

I wrote `numerics` to keep the code from my research and my numerical methods courses in one
place. I maintain it alone and it changes as my research does, so use it with caution!!

Each feature below has an example in [`examples/`](reference/examples/index.md), a complete
program shown with its output.

## Setup

```cmake
include(FetchContent)
FetchContent_Declare(numerics
    GIT_REPOSITORY https://github.com/AdityaDendukuri/numerics.git
    GIT_TAG v1.0.0)
FetchContent_MakeAvailable(numerics)
target_link_libraries(my_program PRIVATE numerics::numerics)
```

| Target | Contents |
| :--- | :--- |
| `numerics::numerics` | Everything, with the backends the build found |
| `numerics::core` | Containers, concepts and portable kernels, with no dependencies |
| `numerics::kernel` | Just `num::kernel` |
| `numerics::blas`, `lapack`, `openmp`, `fftw`, `suitesparse`, `simd`, `mpi`, `cuda` | One backend each |

`#include <numerics.hpp>` includes the whole library.

## From my PhD research

| Feature | Example |
| :--- | :--- |
| Refactoring after a local change. Block LU and block Cholesky keep every block before the change, and no-pivot LU is corrected by Woodbury while few states differ. | [16_factor_reuse](reference/examples/16_factor_reuse.md) |
| Solving with $A + PQ^T$ from a factorization of $A$, including transposed solves and the updated diagonal of the inverse | [17_woodbury_low_rank_update](reference/examples/17_woodbury_low_rank_update.md) |
| $\operatorname{diag}(A^{-1})$ of a sparse M-matrix from a block of random probes | [19_probed_inverse_diagonal](reference/examples/19_probed_inverse_diagonal.md) |
| $(sI - A)x = b$ at many shifts from one Hessenberg reduction, and inverse Laplace transforms by Talbot quadrature | [03_resolvent_and_expv](reference/examples/03_resolvent_and_expv.md), [13_talbot_spectral_validation](reference/examples/13_talbot_spectral_validation.md) |
| The approximate Cholesky factorization of Gao, Kyng and Spielman, as a preconditioner for graph Laplacians | [21_graphs_and_structures](reference/examples/21_graphs_and_structures.md) |

## Feature index

| Area | Features | Example |
| :--- | :--- | :--- |
| Containers | `vec<T>`, `mat<T>`, `spmat`, `num::ops` expressions, in-place `matvec`, `axpy`, selection helpers | [00](reference/examples/00_core_storage_and_helpers.md) |
| Direct solvers | `lu`, `lu(A, no_pivot)`, `cholesky`, `qr` least squares, `thomas`, `det`, `inverse`, `rcond`, one `solve(F, b, x)` for all | [01](reference/examples/01_direct_factorizations.md) |
| Iterative solvers | `cg`, `pcg` with Jacobi, `minres`, `gmres`, sparse and matrix-free operators | [02](reference/examples/02_iterative_krylov_solvers.md) |
| Shifted systems | `hessenberg_resolvent`, `shift`, `solve_batch`, `talbot_contour`, `expv` | [03](reference/examples/03_resolvent_and_expv.md), [12](reference/examples/12_hessenberg_resolvent_benchmark.md), [13](reference/examples/13_talbot_spectral_validation.md) |
| Eigenvalues | `eig_sym`, `lanczos`, `power_iteration`, `svd` | [04](reference/examples/04_eigen_and_svd.md) |
| ODEs | `ode_rk4`, `ode_rk45`, `ode_verlet`, `ode_yoshida4` | [05](reference/examples/05_symplectic_nbody_ode.md) |
| PDEs | 3D fields, `field_solver::solve_poisson`, backward-Euler operators that carry the SPD law | [06](reference/examples/06_pde_poisson_solver.md), [15](reference/examples/15_diffusion_evidence_cg.md) |
| Spectral | `fft`, `ifft`, `rfft`, `irfft`, backends | [07](reference/examples/07_spectral_fft_transforms.md) |
| Roots and quadrature | `bisection`, `newton`, `secant`, `brent`, `trapz`, `simpson`, `gauss_legendre`, `adaptive_simpson`, `romberg` | [08](reference/examples/08_root_finding_and_quadrature.md) |
| Statistics | `running_stats`, `histogram` | [09](reference/examples/09_mcmc_bayesian_sampling.md) |
| Banded matrices | `band_mat`, banded LU with pivoting, many right-hand sides | [10](reference/examples/10_banded_and_spd_operators.md) |
| Plotting | gnuplot figures, plots in the terminal | [11](reference/examples/11_terminal_ascii_plot.md) |
| Laws | `assume_spd`, `make_spd`, compile-time requirements, `num::unsafe` | [14](reference/examples/14_concepts_and_property_invariants.md), [15](reference/examples/15_diffusion_evidence_cg.md) |
| Factor reuse | `lu(R, no_pivot, previous, changed)`, `lu(R, blocks(levels), previous, changed)`, `cholesky(R, blocks(levels), h, previous, changed)` | [16](reference/examples/16_factor_reuse.md) |
| Low-rank updates | `low_rank_difference`, `woodbury_solver`, `transpose(F)`, the updated inverse diagonal | [17](reference/examples/17_woodbury_low_rank_update.md) |
| Precision and conditioning | `rcond`, `inverse_norm1_estimate`, `lu(A, mixed_precision)` | [18](reference/examples/18_mixed_precision_and_conditioning.md) |
| Inverse diagonal | `inverse_diagonal(F, A)` | [19](reference/examples/19_probed_inverse_diagonal.md) |
| Backends and kernel | `num::seq`, `blas`, `omp`, `lapack`, `num::kernel` on raw pointers | [20](reference/examples/20_backends_and_kernel.md) |
| Graphs | `graph`, `dijkstra`, `minimum_spanning_tree`, `disjoint_set`, Laplacian PCG with ApproxChol | [21](reference/examples/21_graphs_and_structures.md) |

## Reference

| Page | Contents |
| :--- | :--- |
| [Containers](reference/containers.md) | Containers and their operations |
| [Concepts](reference/concepts.md) | Concepts and laws |
| [Kernels](reference/kernel.md) | `num::kernel`, the BLAS layer |
| [Linear algebra](reference/linear_algebra.md) | Factorizations, solvers, eigenvalues, the SVD |
| [ODEs](reference/odes.md) | Integrators and their step ranges |
| [PDEs and fields](reference/pdes_and_fields.md) | Grids, fields, stencils, diffusion, Poisson |
| [Spectral](reference/spectral_transforms.md) | FFT and sine transforms |
| [Quadrature and roots](reference/quadrature_and_roots.md) | Integration rules and root finding |
| [Statistics and sampling](reference/statistics.md) | Running statistics, random engines, MCMC |
| [Data structures](reference/data_structures.md) | Graphs, queues, union-find, cell lists |
| [Algorithms](reference/algorithms.md) | BFS, DFS, Dijkstra, spanning trees, generators |
| [All names](reference/index.md) | Every public name, by directory |
| [Benchmark report](report/REPORT.md) | Measured performance |
