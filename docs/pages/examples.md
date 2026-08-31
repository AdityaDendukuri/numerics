# Feature Examples {#page_examples}

Complete domain guides and runnable feature examples organized by topic.

---

## Foundations & Linear Algebra

- @subpage page_container "Vectors, Matrices & Sparse Storage" — Memory layouts, BLAS/SIMD operations, slicing, and views.
- @subpage page_linear "Linear Solvers & Decompositions" — Direct factorizations (LU, Cholesky, QR), Krylov methods (CG, GMRES, MINRES, BiCGSTAB), eigensystems, and SVD.
- @subpage page_operator "Matrix-Free Linear Operators" — Non-owning dense/sparse adapters, lambda stencils, and subspace projectors.
- @subpage page_structures "Discrete Structures & Graph Algorithms" — Disjoint-set union-find, indexed priority queues, degree queues, and graph generators.
- @subpage page_solve "Unified Problem Dispatch" — Façade descriptors (`LinearProblem`, `ODEProblem`) and automated solver dispatch.
- @subpage page_algebra "Algebraic Hierarchy & Field Laws" — Scalar fields, vector spaces, and linear operator property tags.

---

## Numerical Methods

- @subpage page_ode "Ordinary Differential Equations" — Forward Euler, classical RK4, adaptive RK45 Dormand–Prince, and symplectic Verlet/Yoshida integrators.
- @subpage page_pde "Partial Differential Equations" — 2D finite difference stencils, sparse Laplacian assembly, and fast direct Poisson solvers.
- @subpage page_spectral "Fast Fourier Transforms" — Complex FFT/IFFT, real RFFT/IRFFT, Discrete Sine Transforms (DST-I), and plan execution.
- @subpage page_quadrature "Quadrature & Numerical Integration" — Composite trapezoid/Simpson, Gauss–Legendre, adaptive Simpson, Romberg, and Talbot contour integration.
- @subpage page_roots "Root Finding" — Bisection, Brent, Secant, and Newton–Raphson nonlinear scalar solvers.
- @subpage page_stats "Streaming Statistics" — Welford single-pass moments (mean, variance, standard error) and empirical density histograms.
- @subpage page_stochastic "Stochastic & MCMC Sampling" — Categorical sampling, Metropolis–Hastings sweeps, Boltzmann tables, and Umbrella sampling.

---

## Fields, Grids & Spatial Utilities

- @subpage page_fields "Scalar & Vector Fields" — 2D/3D regular grids, trilinear interpolation, differential field operators (grad, div, curl), and magnetic solvers.
- @subpage page_spatial "Spatial Acceleration" — Cell lists, Verlet neighbour lists, and periodic boundary conditions.
- @subpage page_sph_kernel "SPH Smoothing Kernels" — Cubic spline kernel and radial gradient evaluations.
- @subpage page_pbc_lattice "Periodic Boundary Indexing" — Fast modulo-free lattice neighbour lookups.
- @subpage page_connected_components "Connected-Component Labeling" — Spatial component partitioning.
- @subpage page_stencil_hof "Higher-Order Stencils" — Matrix-free higher-order discrete operators.
- @subpage page_boltzmann_table "Boltzmann Energy Tables" — Precomputed discrete state transitions.

---

## Application Walkthroughs

- @subpage page_poisson "2D Poisson Solver" — Direct spectral elliptic solver with discrete sine transform.
- @subpage page_heat_demo "2D Heat Equation" — Explicit and implicit parabolic time stepping.
- @subpage page_ns_perf "Incompressible Navier–Stokes Projection" — Pressure Poisson projection step.

