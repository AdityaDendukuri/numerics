/*
 @licstart  The following is the entire license notice for the JavaScript code in this file.

 The MIT License (MIT)

 Copyright (C) 1997-2020 by Dimitri van Heesch

 Permission is hereby granted, free of charge, to any person obtaining a copy of this software
 and associated documentation files (the "Software"), to deal in the Software without restriction,
 including without limitation the rights to use, copy, modify, merge, publish, distribute,
 sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in all copies or
 substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
 BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
 DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

 @licend  The above is the entire license notice for the JavaScript code in this file
*/
var NAVTREE =
[
  [ "numerics", "index.html", [
    [ "Numerics Library", "index.html", "index" ],
    [ "Boltzmann Acceptance Table", "page_boltzmann_table.html", [
      [ "Motivation", "page_boltzmann_table.html#autotoc_md17", null ],
      [ "API", "page_boltzmann_table.html#autotoc_md19", null ],
      [ "Usage", "page_boltzmann_table.html#autotoc_md21", [
        [ "Direct replacement in Ising", "page_boltzmann_table.html#autotoc_md22", null ],
        [ "Building a flat lookup table", "page_boltzmann_table.html#autotoc_md23", null ]
      ] ]
    ] ],
    [ "Conjugate Gradient and the Thomas Algorithm", "page_cg_notes.html", [
      [ "Part I – Conjugate Gradient (CG)", "page_cg_notes.html#autotoc_md24", [
        [ "Problem Statement", "page_cg_notes.html#autotoc_md25", null ],
        [ "Algorithm", "page_cg_notes.html#autotoc_md26", null ],
        [ "Convergence", "page_cg_notes.html#autotoc_md27", null ],
        [ "Preconditioned CG (PCG)", "page_cg_notes.html#autotoc_md28", null ],
        [ "Matrix-Free CG", "page_cg_notes.html#autotoc_md29", null ]
      ] ],
      [ "Part II – Thomas Algorithm (Tridiagonal Solver)", "page_cg_notes.html#autotoc_md31", [
        [ "Problem Statement", "page_cg_notes.html#autotoc_md32", null ],
        [ "Algorithm", "page_cg_notes.html#autotoc_md33", null ],
        [ "Applicability", "page_cg_notes.html#autotoc_md34", null ]
      ] ],
      [ "Performance Optimization", "page_cg_notes.html#autotoc_md36", [
        [ "CG: SIMD for Dense Matvec", "page_cg_notes.html#autotoc_md37", null ],
        [ "CG: Pipelining (Chronopoulos-Gear)", "page_cg_notes.html#autotoc_md38", null ],
        [ "Thomas: SIMD over Multiple RHS", "page_cg_notes.html#autotoc_md39", null ]
      ] ]
    ] ],
    [ "Connected Components", "page_connected_components.html", [
      [ "Motivation", "page_connected_components.html#autotoc_md41", null ],
      [ "API", "page_connected_components.html#autotoc_md43", [
        [ "Labels", "page_connected_components.html#autotoc_md44", null ]
      ] ],
      [ "Algorithm", "page_connected_components.html#autotoc_md46", null ],
      [ "Usage – Ising nucleation", "page_connected_components.html#autotoc_md48", null ],
      [ "Generalizations", "page_connected_components.html#autotoc_md50", null ]
    ] ],
    [ "Symmetric Jacobi Eigendecomposition (eig_sym)", "page_eig_jacobi_notes.html", [
      [ "Problem Statement", "page_eig_jacobi_notes.html#autotoc_md51", null ],
      [ "Mathematical Foundation", "page_eig_jacobi_notes.html#autotoc_md53", [
        [ "Givens Rotation in the \\f$(p,q)\\f$ Plane", "page_eig_jacobi_notes.html#autotoc_md54", null ],
        [ "Choosing \\f$\\theta\\f$ to Zero \\f$A'_{pq}\\f$", "page_eig_jacobi_notes.html#autotoc_md55", null ]
      ] ],
      [ "Algorithm: Cyclic Jacobi", "page_eig_jacobi_notes.html#autotoc_md57", [
        [ "Convergence", "page_eig_jacobi_notes.html#autotoc_md58", null ],
        [ "Monotonicity of Diagonal Entries", "page_eig_jacobi_notes.html#autotoc_md59", null ]
      ] ],
      [ "Variants", "page_eig_jacobi_notes.html#autotoc_md61", [
        [ "Threshold Jacobi", "page_eig_jacobi_notes.html#autotoc_md62", null ],
        [ "One-Sided Jacobi", "page_eig_jacobi_notes.html#autotoc_md63", null ],
        [ "Parallel Jacobi (Tournament Ordering)", "page_eig_jacobi_notes.html#autotoc_md64", null ]
      ] ],
      [ "Comparison with Other Eigensolvers", "page_eig_jacobi_notes.html#autotoc_md66", null ],
      [ "Performance Optimization", "page_eig_jacobi_notes.html#autotoc_md68", [
        [ "Current Implementation", "page_eig_jacobi_notes.html#autotoc_md69", null ],
        [ "SIMD for the Rotation Update (Column-Major A)", "page_eig_jacobi_notes.html#autotoc_md70", null ],
        [ "Exploiting Symmetry in the Update", "page_eig_jacobi_notes.html#autotoc_md71", null ],
        [ "Blocked Jacobi (Brent-Luk)", "page_eig_jacobi_notes.html#autotoc_md72", null ],
        [ "Cache Behaviour", "page_eig_jacobi_notes.html#autotoc_md73", null ]
      ] ]
    ] ],
    [ "Power Iteration, Inverse Iteration, and Rayleigh Quotient Iteration", "page_eig_power_notes.html", [
      [ "Background", "page_eig_power_notes.html#autotoc_md74", null ],
      [ "Part I – Power Iteration", "page_eig_power_notes.html#autotoc_md76", [
        [ "Algorithm", "page_eig_power_notes.html#autotoc_md77", null ],
        [ "Convergence Analysis", "page_eig_power_notes.html#autotoc_md78", null ],
        [ "Failure Modes", "page_eig_power_notes.html#autotoc_md79", null ],
        [ "Deflation (Hotelling)", "page_eig_power_notes.html#autotoc_md80", null ]
      ] ],
      [ "Part II – Inverse Iteration", "page_eig_power_notes.html#autotoc_md82", [
        [ "Algorithm", "page_eig_power_notes.html#autotoc_md83", null ],
        [ "Convergence", "page_eig_power_notes.html#autotoc_md84", null ],
        [ "Use Cases", "page_eig_power_notes.html#autotoc_md85", null ]
      ] ],
      [ "Part III – Rayleigh Quotient Iteration (RQI)", "page_eig_power_notes.html#autotoc_md87", [
        [ "Algorithm", "page_eig_power_notes.html#autotoc_md88", null ],
        [ "Convergence: Cubic Rate", "page_eig_power_notes.html#autotoc_md89", null ],
        [ "Practical Strategy: Combine Methods", "page_eig_power_notes.html#autotoc_md90", null ]
      ] ],
      [ "Performance Optimization", "page_eig_power_notes.html#autotoc_md92", [
        [ "SIMD for Power Iteration Matvec", "page_eig_power_notes.html#autotoc_md93", null ],
        [ "Subspace Iteration (Block Power Method)", "page_eig_power_notes.html#autotoc_md94", null ],
        [ "Blocked Inverse Iteration for Multiple Eigenvectors", "page_eig_power_notes.html#autotoc_md95", null ],
        [ "Shift Selection via Gershgorin Discs", "page_eig_power_notes.html#autotoc_md96", null ]
      ] ]
    ] ],
    [ "3D Field Types and Solvers", "page_fields.html", [
      [ "Types", "page_fields.html#autotoc_md148", [
        [ "ScalarField3D", "page_fields.html#autotoc_md149", null ],
        [ "VectorField3D", "page_fields.html#autotoc_md151", null ]
      ] ],
      [ "FieldSolver", "page_fields.html#autotoc_md153", null ],
      [ "MagneticSolver", "page_fields.html#autotoc_md155", null ],
      [ "Typical Workflow", "page_fields.html#autotoc_md157", null ],
      [ "Where Each Type Is Used", "page_fields.html#autotoc_md159", null ]
    ] ],
    [ "GMRES -- Generalized Minimal Residual Method", "page_gmres_notes.html", [
      [ "Problem Statement", "page_gmres_notes.html#autotoc_md160", null ],
      [ "Arnoldi Process", "page_gmres_notes.html#autotoc_md162", null ],
      [ "The GMRES Least-Squares Problem", "page_gmres_notes.html#autotoc_md164", null ],
      [ "Algorithm: Restarted GMRES(\\f$m\\f$)", "page_gmres_notes.html#autotoc_md166", null ],
      [ "Convergence", "page_gmres_notes.html#autotoc_md168", [
        [ "Preconditioning", "page_gmres_notes.html#autotoc_md169", null ]
      ] ],
      [ "Variants", "page_gmres_notes.html#autotoc_md171", null ],
      [ "Performance Optimization", "page_gmres_notes.html#autotoc_md173", [
        [ "BLAS-2 Arnoldi", "page_gmres_notes.html#autotoc_md174", null ],
        [ "Block Arnoldi (BLAS-3)", "page_gmres_notes.html#autotoc_md175", null ],
        [ "Communication-Avoiding GMRES (CA-GMRES)", "page_gmres_notes.html#autotoc_md176", null ],
        [ "SIMD for Dense Matvec and MGS", "page_gmres_notes.html#autotoc_md177", null ]
      ] ]
    ] ],
    [ "Lanczos Algorithm with Full Reorthogonalization", "page_lanczos_notes.html", [
      [ "Problem Statement", "page_lanczos_notes.html#autotoc_md224", null ],
      [ "The Lanczos Recurrence", "page_lanczos_notes.html#autotoc_md226", null ],
      [ "Algorithm: Lanczos with Full Reorthogonalization (MGS)", "page_lanczos_notes.html#autotoc_md228", null ],
      [ "The Projection Principle", "page_lanczos_notes.html#autotoc_md230", null ],
      [ "Convergence", "page_lanczos_notes.html#autotoc_md232", null ],
      [ "Ghost Eigenvalues and Reorthogonalization", "page_lanczos_notes.html#autotoc_md234", null ],
      [ "Performance Optimization", "page_lanczos_notes.html#autotoc_md236", [
        [ "BLAS-2 Reorthogonalization", "page_lanczos_notes.html#autotoc_md237", null ],
        [ "Block Lanczos (BLAS-3)", "page_lanczos_notes.html#autotoc_md238", null ],
        [ "Sparse Matrix Formats for SpMV", "page_lanczos_notes.html#autotoc_md239", null ],
        [ "Distributed Lanczos (MPI)", "page_lanczos_notes.html#autotoc_md240", null ]
      ] ]
    ] ],
    [ "LU Factorization with Partial Pivoting", "page_lu_notes.html", [
      [ "Overview", "page_lu_notes.html#autotoc_md267", null ],
      [ "Algorithm", "page_lu_notes.html#autotoc_md269", null ],
      [ "Forward/Backward Substitution", "page_lu_notes.html#autotoc_md271", null ],
      [ "Determinant and Inverse", "page_lu_notes.html#autotoc_md273", null ],
      [ "Stability", "page_lu_notes.html#autotoc_md275", null ],
      [ "Performance Optimization", "page_lu_notes.html#autotoc_md277", [
        [ "Current Implementation", "page_lu_notes.html#autotoc_md278", null ],
        [ "Level-3 BLAS: Blocked LU (LAPACK <tt>dgetrf</tt>)", "page_lu_notes.html#autotoc_md279", null ],
        [ "SIMD for the Schur Complement", "page_lu_notes.html#autotoc_md280", null ],
        [ "Recursive LU", "page_lu_notes.html#autotoc_md281", null ],
        [ "Multiple Right-Hand Sides", "page_lu_notes.html#autotoc_md282", null ]
      ] ],
      [ "Relation to Other Factorizations", "page_lu_notes.html#autotoc_md284", null ]
    ] ],
    [ "NS Demo: From Slideshow to Real-Time", "page_ns_perf.html", [
      [ "The Solver in Brief", "page_ns_perf.html#sec_ns_overview", null ],
      [ "The Periodic Laplacian Stencil", "page_ns_perf.html#sec_stencil", null ],
      [ "Fix 1 – Boundary Peeling for NEON Auto-Vectorisation", "page_ns_perf.html#sec_fix_vectorise", [
        [ "The Problem", "page_ns_perf.html#autotoc_md288", null ],
        [ "The Fix", "page_ns_perf.html#autotoc_md289", null ],
        [ "Arithmetic Intensity", "page_ns_perf.html#autotoc_md290", null ]
      ] ],
      [ "Fix 2 – <tt>Backend::blas</tt> over <tt>Backend::omp</tt> for Cache-Resident Vectors", "page_ns_perf.html#sec_fix_policy", [
        [ "The Problem", "page_ns_perf.html#autotoc_md292", null ],
        [ "The Fix", "page_ns_perf.html#autotoc_md293", null ],
        [ "Backend Crossover Rule of Thumb", "page_ns_perf.html#autotoc_md294", null ]
      ] ],
      [ "Fix 3 – CG Tolerance and Warm-Starting", "page_ns_perf.html#sec_fix_cg", [
        [ "Spectral Condition Number of the Periodic Laplacian", "page_ns_perf.html#autotoc_md296", null ],
        [ "CG Convergence Rate", "page_ns_perf.html#autotoc_md297", null ],
        [ "Warm-Starting", "page_ns_perf.html#autotoc_md298", null ],
        [ "Periodic Poisson Singularity", "page_ns_perf.html#autotoc_md299", null ]
      ] ],
      [ "Combined Effect", "page_ns_perf.html#sec_ns_combined", null ],
      [ "Building and Running", "page_ns_perf.html#sec_ns_build", null ]
    ] ],
    [ "ODE Module", "page_ode.html", [
      [ "Types", "page_ode.html#autotoc_md303", null ],
      [ "Usage patterns", "page_ode.html#autotoc_md305", [
        [ "Final state only", "page_ode.html#autotoc_md306", null ],
        [ "Observe every step", "page_ode.html#autotoc_md307", null ]
      ] ],
      [ "Integrators", "page_ode.html#autotoc_md309", [
        [ "Explicit Euler – O(h)", "page_ode.html#autotoc_md310", null ],
        [ "Classic RK4 – O(h^4)", "page_ode.html#autotoc_md311", null ],
        [ "Adaptive RK45 – O(h^5) with error control", "page_ode.html#autotoc_md312", null ],
        [ "Velocity Verlet – O(h^2), symplectic", "page_ode.html#autotoc_md313", null ],
        [ "Yoshida 4th-order – O(h^4), symplectic", "page_ode.html#autotoc_md314", null ],
        [ "RK4 for second-order systems (Nystrom form)", "page_ode.html#autotoc_md315", null ]
      ] ],
      [ "Symplectic vs. RK4: Energy Conservation", "page_ode.html#autotoc_md317", null ],
      [ "Example: Lorenz attractor", "page_ode.html#autotoc_md319", null ],
      [ "Example: Kepler orbit", "page_ode.html#autotoc_md320", null ],
      [ "Tests", "page_ode.html#autotoc_md322", null ],
      [ "References", "page_ode.html#autotoc_md324", null ]
    ] ],
    [ "PBCLattice2D", "page_pbc_lattice.html", [
      [ "Motivation", "page_pbc_lattice.html#autotoc_md347", null ],
      [ "API", "page_pbc_lattice.html#autotoc_md349", null ],
      [ "Index Layout", "page_pbc_lattice.html#autotoc_md351", null ],
      [ "Usage", "page_pbc_lattice.html#autotoc_md353", null ]
    ] ],
    [ "PDE Module", "page_pde.html", [
      [ "Contents", "page_pde.html#autotoc_md355", null ],
      [ "Stencil operators (<tt>pde/stencil.hpp</tt>)", "page_pde.html#autotoc_md357", [
        [ "2D operators", "page_pde.html#autotoc_md358", null ],
        [ "2D grid utilities", "page_pde.html#autotoc_md359", null ],
        [ "2D fiber sweeps", "page_pde.html#autotoc_md360", null ],
        [ "3D operators", "page_pde.html#autotoc_md361", null ]
      ] ],
      [ "3D fields (<tt>pde/fields.hpp</tt>)", "page_pde.html#autotoc_md363", null ],
      [ "Crank-Nicolson ADI (<tt>pde/adi.hpp</tt>)", "page_pde.html#autotoc_md365", [
        [ "Sweep", "page_pde.html#autotoc_md366", null ],
        [ "Strang splitting (TDSE)", "page_pde.html#autotoc_md367", null ]
      ] ],
      [ "Implicit backward Euler (<tt>pde/diffusion.hpp</tt>)", "page_pde.html#autotoc_md369", null ],
      [ "Explicit diffusion (<tt>pde/stencil.hpp</tt>)", "page_pde.html#autotoc_md370", [
        [ "Typical usage (Navier-Stokes viscosity)", "page_pde.html#autotoc_md371", null ]
      ] ],
      [ "Backward compatibility", "page_pde.html#autotoc_md373", null ]
    ] ],
    [ "QR Factorization via Householder Reflections", "page_qr_notes.html", [
      [ "Overview", "page_qr_notes.html#autotoc_md399", null ],
      [ "Householder Reflections", "page_qr_notes.html#autotoc_md401", null ],
      [ "Algorithm", "page_qr_notes.html#autotoc_md403", null ],
      [ "Least-Squares Solve", "page_qr_notes.html#autotoc_md405", null ],
      [ "Stability", "page_qr_notes.html#autotoc_md407", null ],
      [ "Alternatives: Givens vs MGS vs Householder", "page_qr_notes.html#autotoc_md409", null ],
      [ "Performance Optimization", "page_qr_notes.html#autotoc_md411", [
        [ "Current Implementation", "page_qr_notes.html#autotoc_md412", null ],
        [ "Blocked Householder: WY Representation", "page_qr_notes.html#autotoc_md413", null ],
        [ "SIMD for the AXPY Kernel", "page_qr_notes.html#autotoc_md414", null ],
        [ "Avoiding Explicit Q", "page_qr_notes.html#autotoc_md415", null ],
        [ "Tall-Skinny QR (TSQR) for Distributed Systems", "page_qr_notes.html#autotoc_md416", null ]
      ] ]
    ] ],
    [ "Numerical Quadrature", "page_quadrature_notes.html", [
      [ "Newton-Cotes Rules", "page_quadrature_notes.html#autotoc_md418", [
        [ "Trapezoidal Rule", "page_quadrature_notes.html#autotoc_md419", null ],
        [ "Simpson's Rule", "page_quadrature_notes.html#autotoc_md420", null ]
      ] ],
      [ "Gaussian Quadrature", "page_quadrature_notes.html#autotoc_md422", null ],
      [ "Adaptive Quadrature", "page_quadrature_notes.html#autotoc_md424", null ],
      [ "Romberg Integration", "page_quadrature_notes.html#autotoc_md426", null ],
      [ "Comparison", "page_quadrature_notes.html#autotoc_md428", null ],
      [ "Parallel Structure", "page_quadrature_notes.html#autotoc_md430", [
        [ "Domain decomposition", "page_quadrature_notes.html#autotoc_md431", null ],
        [ "Adaptive load balancing", "page_quadrature_notes.html#autotoc_md432", null ],
        [ "Gauss-Legendre with many panels", "page_quadrature_notes.html#autotoc_md433", null ]
      ] ],
      [ "Exercises", "page_quadrature_notes.html#autotoc_md435", null ]
    ] ],
    [ "Root Finding", "page_roots_notes.html", [
      [ "Bisection", "page_roots_notes.html#autotoc_md437", null ],
      [ "Newton-Raphson", "page_roots_notes.html#autotoc_md439", null ],
      [ "Secant Method", "page_roots_notes.html#autotoc_md441", null ],
      [ "Brent's Method", "page_roots_notes.html#autotoc_md443", null ],
      [ "Comparison", "page_roots_notes.html#autotoc_md445", null ],
      [ "Parallel Structure", "page_roots_notes.html#autotoc_md447", null ],
      [ "Exercises", "page_roots_notes.html#autotoc_md449", null ]
    ] ],
    [ "SPH Kernels", "page_sph_kernel.html", [
      [ "Motivation", "page_sph_kernel.html#autotoc_md451", null ],
      [ "API", "page_sph_kernel.html#autotoc_md453", null ],
      [ "Kernels", "page_sph_kernel.html#autotoc_md455", [
        [ "Cubic Spline – density", "page_sph_kernel.html#autotoc_md456", null ],
        [ "Spiky Kernel – pressure gradient", "page_sph_kernel.html#autotoc_md457", null ]
      ] ],
      [ "Dimension specialization", "page_sph_kernel.html#autotoc_md459", null ],
      [ "Usage", "page_sph_kernel.html#autotoc_md461", null ],
      [ "Where Each Kernel Is Used", "page_sph_kernel.html#autotoc_md463", null ]
    ] ],
    [ "Stationary Iterative Solvers: Jacobi and Gauss-Seidel", "page_stationary_notes.html", [
      [ "Problem Statement", "page_stationary_notes.html#autotoc_md464", null ],
      [ "Part I – Jacobi Iteration", "page_stationary_notes.html#autotoc_md466", [
        [ "Algorithm", "page_stationary_notes.html#autotoc_md467", null ],
        [ "Convergence Conditions", "page_stationary_notes.html#autotoc_md468", null ],
        [ "Jacobi as a Multigrid Smoother", "page_stationary_notes.html#autotoc_md469", null ]
      ] ],
      [ "Part II – Gauss-Seidel Iteration", "page_stationary_notes.html#autotoc_md471", [
        [ "Algorithm", "page_stationary_notes.html#autotoc_md472", null ],
        [ "Gauss-Seidel vs Jacobi", "page_stationary_notes.html#autotoc_md473", null ],
        [ "SOR – Successive Over-Relaxation", "page_stationary_notes.html#autotoc_md474", null ]
      ] ],
      [ "Ordering and Parallelism", "page_stationary_notes.html#autotoc_md476", [
        [ "Red-Black Ordering", "page_stationary_notes.html#autotoc_md477", null ],
        [ "Multi-Color Ordering", "page_stationary_notes.html#autotoc_md478", null ]
      ] ],
      [ "Performance Optimization", "page_stationary_notes.html#autotoc_md480", [
        [ "SIMD for the Row Dot Product", "page_stationary_notes.html#autotoc_md481", null ],
        [ "Cache Blocking for Large n", "page_stationary_notes.html#autotoc_md482", null ],
        [ "Sparse Gauss-Seidel (CSR Format)", "page_stationary_notes.html#autotoc_md483", null ],
        [ "Parallelism Summary", "page_stationary_notes.html#autotoc_md484", null ]
      ] ]
    ] ],
    [ "Stencil Higher-Order Functions", "page_stencil_hof.html", [
      [ "Motivation", "page_stencil_hof.html#autotoc_md486", null ],
      [ "2D Functions", "page_stencil_hof.html#autotoc_md488", [
        [ "laplacian_stencil_2d – Dirichlet boundaries", "page_stencil_hof.html#autotoc_md489", null ],
        [ "laplacian_stencil_2d_periodic – Periodic boundaries", "page_stencil_hof.html#autotoc_md491", null ],
        [ "col_fiber_sweep / row_fiber_sweep – ADI/CN sweeps", "page_stencil_hof.html#autotoc_md493", null ]
      ] ],
      [ "3D Functions", "page_stencil_hof.html#autotoc_md495", [
        [ "neg_laplacian_3d – 7-point stencil", "page_stencil_hof.html#autotoc_md496", null ],
        [ "gradient_3d", "page_stencil_hof.html#autotoc_md498", null ],
        [ "divergence_3d", "page_stencil_hof.html#autotoc_md500", null ],
        [ "curl_3d", "page_stencil_hof.html#autotoc_md502", null ]
      ] ],
      [ "Where Each Function Is Used", "page_stencil_hof.html#autotoc_md504", null ],
      [ "Adding a New App", "page_stencil_hof.html#autotoc_md506", null ]
    ] ],
    [ "One-Sided Jacobi SVD", "page_svd_jacobi_notes.html", [
      [ "Problem Statement", "page_svd_jacobi_notes.html#autotoc_md507", null ],
      [ "Mathematical Foundation", "page_svd_jacobi_notes.html#autotoc_md509", [
        [ "Relation to the Eigenvalue Problem", "page_svd_jacobi_notes.html#autotoc_md510", null ],
        [ "The Column Orthogonality Condition", "page_svd_jacobi_notes.html#autotoc_md511", null ],
        [ "Rotation Angle to Zero \\f$[A^TA]_{pq}\\f$", "page_svd_jacobi_notes.html#autotoc_md512", null ]
      ] ],
      [ "Algorithm", "page_svd_jacobi_notes.html#autotoc_md514", null ],
      [ "Convergence Analysis", "page_svd_jacobi_notes.html#autotoc_md516", null ],
      [ "Accuracy Properties", "page_svd_jacobi_notes.html#autotoc_md518", null ],
      [ "Comparison with Alternatives", "page_svd_jacobi_notes.html#autotoc_md520", null ],
      [ "Performance Optimization", "page_svd_jacobi_notes.html#autotoc_md522", [
        [ "Current Implementation", "page_svd_jacobi_notes.html#autotoc_md523", null ],
        [ "SIMD for Column Inner Products and Rotation", "page_svd_jacobi_notes.html#autotoc_md524", null ],
        [ "Fused Single-Pass Update", "page_svd_jacobi_notes.html#autotoc_md525", null ],
        [ "Batched Rotations (Row-Blocked Update)", "page_svd_jacobi_notes.html#autotoc_md526", null ],
        [ "Pre-Processing: Column Equilibration", "page_svd_jacobi_notes.html#autotoc_md527", null ],
        [ "Block One-Sided Jacobi", "page_svd_jacobi_notes.html#autotoc_md528", null ]
      ] ]
    ] ],
    [ "Randomized Truncated SVD", "page_svd_random_notes.html", [
      [ "Problem Statement", "page_svd_random_notes.html#autotoc_md529", null ],
      [ "Key Insight: Randomized Range Finding", "page_svd_random_notes.html#autotoc_md531", null ],
      [ "Algorithm", "page_svd_random_notes.html#autotoc_md533", null ],
      [ "Step-by-Step Analysis", "page_svd_random_notes.html#autotoc_md535", [
        [ "Step 1: Random Sketch", "page_svd_random_notes.html#autotoc_md536", null ],
        [ "Step 2: Matrix-Vector Product \\f$Y = A\\Omega\\f$", "page_svd_random_notes.html#autotoc_md537", null ],
        [ "Step 3: QR of \\f$Y\\f$", "page_svd_random_notes.html#autotoc_md538", null ],
        [ "Step 4: Projection \\f$B = Q^TA\\f$", "page_svd_random_notes.html#autotoc_md539", null ],
        [ "Step 5: Small SVD of \\f$B\\f$", "page_svd_random_notes.html#autotoc_md540", null ],
        [ "Step 6: Lift to Full Space \\f$U = Q\\hat{U}_\\ell\\f$", "page_svd_random_notes.html#autotoc_md541", null ]
      ] ],
      [ "Oversampling Cap: Why \\f$\\ell \\leq \\min(m,n)\\f$", "page_svd_random_notes.html#autotoc_md543", null ],
      [ "Accuracy vs Cost Trade-off", "page_svd_random_notes.html#autotoc_md545", null ],
      [ "Applications", "page_svd_random_notes.html#autotoc_md547", null ],
      [ "Performance Optimization", "page_svd_random_notes.html#autotoc_md549", [
        [ "Current Implementation", "page_svd_random_notes.html#autotoc_md550", null ],
        [ "Streaming Computation (Out-of-Core)", "page_svd_random_notes.html#autotoc_md551", null ],
        [ "GPU Implementation", "page_svd_random_notes.html#autotoc_md552", null ],
        [ "Avoiding Step 4 via Single-View Algorithm", "page_svd_random_notes.html#autotoc_md553", null ],
        [ "Structured Sketches for Sparse \\f$A\\f$", "page_svd_random_notes.html#autotoc_md554", null ]
      ] ]
    ] ],
    [ "README", "md__2home_2runner_2work_2numerics_2numerics_2apps_2em__demo_2README.html", [
      [ "Electromagnetic Field Demo", "md__2home_2runner_2work_2numerics_2numerics_2apps_2em__demo_2README.html#autotoc_md881", [
        [ "Physical Model", "md__2home_2runner_2work_2numerics_2numerics_2apps_2em__demo_2README.html#autotoc_md883", [
          [ "Electrostatics – Steady Current Flow", "md__2home_2runner_2work_2numerics_2numerics_2apps_2em__demo_2README.html#autotoc_md884", null ],
          [ "Magnetostatics – Vector Potential", "md__2home_2runner_2work_2numerics_2numerics_2apps_2em__demo_2README.html#autotoc_md885", null ]
        ] ],
        [ "Geometry", "md__2home_2runner_2work_2numerics_2numerics_2apps_2em__demo_2README.html#autotoc_md887", null ],
        [ "Numerical Method", "md__2home_2runner_2work_2numerics_2numerics_2apps_2em__demo_2README.html#autotoc_md889", [
          [ "Variable-Coefficient Potential Solve", "md__2home_2runner_2work_2numerics_2numerics_2apps_2em__demo_2README.html#autotoc_md890", null ],
          [ "Magnetic Poisson Solves", "md__2home_2runner_2work_2numerics_2numerics_2apps_2em__demo_2README.html#autotoc_md891", null ]
        ] ]
      ] ]
    ] ],
    [ "README", "md__2home_2runner_2work_2numerics_2numerics_2apps_2fluid__sim_2README.html", [
      [ "2D SPH Fluid Simulation", "md__2home_2runner_2work_2numerics_2numerics_2apps_2fluid__sim_2README.html#autotoc_md901", [
        [ "Physical Model", "md__2home_2runner_2work_2numerics_2numerics_2apps_2fluid__sim_2README.html#autotoc_md903", null ],
        [ "SPH Kernels", "md__2home_2runner_2work_2numerics_2numerics_2apps_2fluid__sim_2README.html#autotoc_md905", [
          [ "Cubic Spline (density)", "md__2home_2runner_2work_2numerics_2numerics_2apps_2fluid__sim_2README.html#autotoc_md906", null ],
          [ "Spiky Kernel (pressure gradient)", "md__2home_2runner_2work_2numerics_2numerics_2apps_2fluid__sim_2README.html#autotoc_md907", null ],
          [ "Morris Laplacian (viscosity, heat)", "md__2home_2runner_2work_2numerics_2numerics_2apps_2fluid__sim_2README.html#autotoc_md908", null ]
        ] ],
        [ "Equation of State", "md__2home_2runner_2work_2numerics_2numerics_2apps_2fluid__sim_2README.html#autotoc_md910", null ],
        [ "Neighbour Search – <tt>CellList2D</tt>", "md__2home_2runner_2work_2numerics_2numerics_2apps_2fluid__sim_2README.html#autotoc_md912", null ],
        [ "Time Integration", "md__2home_2runner_2work_2numerics_2numerics_2apps_2fluid__sim_2README.html#autotoc_md914", null ],
        [ "Numerics Library Integration", "md__2home_2runner_2work_2numerics_2numerics_2apps_2fluid__sim_2README.html#autotoc_md916", null ],
        [ "Project layout", "md__2home_2runner_2work_2numerics_2numerics_2apps_2fluid__sim_2README.html#autotoc_md918", null ],
        [ "Running", "md__2home_2runner_2work_2numerics_2numerics_2apps_2fluid__sim_2README.html#autotoc_md919", null ],
        [ "References", "md__2home_2runner_2work_2numerics_2numerics_2apps_2fluid__sim_2README.html#autotoc_md921", null ]
      ] ]
    ] ],
    [ "README", "md__2home_2runner_2work_2numerics_2numerics_2apps_2fluid__sim__3d_2README.html", [
      [ "3D SPH Fluid Simulation", "md__2home_2runner_2work_2numerics_2numerics_2apps_2fluid__sim__3d_2README.html#autotoc_md922", [
        [ "Physical Model", "md__2home_2runner_2work_2numerics_2numerics_2apps_2fluid__sim__3d_2README.html#autotoc_md924", null ],
        [ "3D Kernels", "md__2home_2runner_2work_2numerics_2numerics_2apps_2fluid__sim__3d_2README.html#autotoc_md926", [
          [ "Cubic Spline", "md__2home_2runner_2work_2numerics_2numerics_2apps_2fluid__sim__3d_2README.html#autotoc_md927", null ],
          [ "Spiky Kernel (pressure gradient)", "md__2home_2runner_2work_2numerics_2numerics_2apps_2fluid__sim__3d_2README.html#autotoc_md928", null ],
          [ "Morris Laplacian", "md__2home_2runner_2work_2numerics_2numerics_2apps_2fluid__sim__3d_2README.html#autotoc_md929", null ]
        ] ],
        [ "Hose Jets", "md__2home_2runner_2work_2numerics_2numerics_2apps_2fluid__sim__3d_2README.html#autotoc_md931", null ],
        [ "Neighbour Search – <tt>CellList3D</tt>", "md__2home_2runner_2work_2numerics_2numerics_2apps_2fluid__sim__3d_2README.html#autotoc_md933", null ],
        [ "Numerics Library Integration", "md__2home_2runner_2work_2numerics_2numerics_2apps_2fluid__sim__3d_2README.html#autotoc_md935", null ],
        [ "Project layout", "md__2home_2runner_2work_2numerics_2numerics_2apps_2fluid__sim__3d_2README.html#autotoc_md937", null ],
        [ "Running", "md__2home_2runner_2work_2numerics_2numerics_2apps_2fluid__sim__3d_2README.html#autotoc_md938", null ],
        [ "References", "md__2home_2runner_2work_2numerics_2numerics_2apps_2fluid__sim__3d_2README.html#autotoc_md940", null ]
      ] ]
    ] ],
    [ "README", "md__2home_2runner_2work_2numerics_2numerics_2apps_2ns__demo_2README.html", [
      [ "2D Incompressible Navier-Stokes Solver", "md__2home_2runner_2work_2numerics_2numerics_2apps_2ns__demo_2README.html#autotoc_md941", [
        [ "Physical Model", "md__2home_2runner_2work_2numerics_2numerics_2apps_2ns__demo_2README.html#autotoc_md943", null ],
        [ "MAC Grid", "md__2home_2runner_2work_2numerics_2numerics_2apps_2ns__demo_2README.html#autotoc_md945", null ],
        [ "Chorin Projection Method", "md__2home_2runner_2work_2numerics_2numerics_2apps_2ns__demo_2README.html#autotoc_md947", [
          [ "1. Semi-Lagrangian Advection", "md__2home_2runner_2work_2numerics_2numerics_2apps_2ns__demo_2README.html#autotoc_md948", null ],
          [ "2. Pressure Poisson Solve", "md__2home_2runner_2work_2numerics_2numerics_2apps_2ns__demo_2README.html#autotoc_md949", null ],
          [ "3. Velocity Projection", "md__2home_2runner_2work_2numerics_2numerics_2apps_2ns__demo_2README.html#autotoc_md950", null ]
        ] ],
        [ "Initial Condition – Double Shear Layer", "md__2home_2runner_2work_2numerics_2numerics_2apps_2ns__demo_2README.html#autotoc_md952", null ],
        [ "Numerics Library Integration", "md__2home_2runner_2work_2numerics_2numerics_2apps_2ns__demo_2README.html#autotoc_md954", null ],
        [ "Visualisation", "md__2home_2runner_2work_2numerics_2numerics_2apps_2ns__demo_2README.html#autotoc_md956", null ],
        [ "Project layout", "md__2home_2runner_2work_2numerics_2numerics_2apps_2ns__demo_2README.html#autotoc_md958", null ],
        [ "Running", "md__2home_2runner_2work_2numerics_2numerics_2apps_2ns__demo_2README.html#autotoc_md959", null ],
        [ "References", "md__2home_2runner_2work_2numerics_2numerics_2apps_2ns__demo_2README.html#autotoc_md960", null ]
      ] ]
    ] ],
    [ "README", "md__2home_2runner_2work_2numerics_2numerics_2apps_2tdse_2README.html", [
      [ "2D Time-Dependent Schrodinger Equation", "md__2home_2runner_2work_2numerics_2numerics_2apps_2tdse_2README.html#autotoc_md961", [
        [ "Preset scene", "md__2home_2runner_2work_2numerics_2numerics_2apps_2tdse_2README.html#autotoc_md963", null ],
        [ "Physical Model", "md__2home_2runner_2work_2numerics_2numerics_2apps_2tdse_2README.html#autotoc_md965", null ],
        [ "Strang Operator Splitting", "md__2home_2runner_2work_2numerics_2numerics_2apps_2tdse_2README.html#autotoc_md967", null ],
        [ "Thomas Algorithm", "md__2home_2runner_2work_2numerics_2numerics_2apps_2tdse_2README.html#autotoc_md969", null ],
        [ "Potentials", "md__2home_2runner_2work_2numerics_2numerics_2apps_2tdse_2README.html#autotoc_md971", null ],
        [ "Eigenmode Computation – Lanczos", "md__2home_2runner_2work_2numerics_2numerics_2apps_2tdse_2README.html#autotoc_md973", null ],
        [ "Observables", "md__2home_2runner_2work_2numerics_2numerics_2apps_2tdse_2README.html#autotoc_md975", null ],
        [ "Numerics Library Integration", "md__2home_2runner_2work_2numerics_2numerics_2apps_2tdse_2README.html#autotoc_md977", null ],
        [ "Visualisation", "md__2home_2runner_2work_2numerics_2numerics_2apps_2tdse_2README.html#autotoc_md979", null ],
        [ "Project layout", "md__2home_2runner_2work_2numerics_2numerics_2apps_2tdse_2README.html#autotoc_md981", null ],
        [ "Running", "md__2home_2runner_2work_2numerics_2numerics_2apps_2tdse_2README.html#autotoc_md982", null ],
        [ "Build", "md__2home_2runner_2work_2numerics_2numerics_2apps_2tdse_2README.html#autotoc_md984", [
          [ "Using CMake presets (recommended)", "md__2home_2runner_2work_2numerics_2numerics_2apps_2tdse_2README.html#autotoc_md985", null ],
          [ "Manual configuration", "md__2home_2runner_2work_2numerics_2numerics_2apps_2tdse_2README.html#autotoc_md986", null ]
        ] ],
        [ "References", "md__2home_2runner_2work_2numerics_2numerics_2apps_2tdse_2README.html#autotoc_md988", null ]
      ] ]
    ] ],
    [ "README", "md__2home_2runner_2work_2numerics_2numerics_2apps_2ising_2README.html", [
      [ "2D Ising Model", "md__2home_2runner_2work_2numerics_2numerics_2apps_2ising_2README.html#autotoc_md989", [
        [ "Preset scene", "md__2home_2runner_2work_2numerics_2numerics_2apps_2ising_2README.html#autotoc_md991", null ],
        [ "Physical Model", "md__2home_2runner_2work_2numerics_2numerics_2apps_2ising_2README.html#autotoc_md993", null ]
      ] ]
    ] ],
    [ "README", "md__2home_2runner_2work_2numerics_2numerics_2apps_2quantum__demo_2README.html", [
      [ "Quantum Circuit Simulator", "md__2home_2runner_2work_2numerics_2numerics_2apps_2quantum__demo_2README.html#autotoc_md1013", [
        [ "Physical Model", "md__2home_2runner_2work_2numerics_2numerics_2apps_2quantum__demo_2README.html#autotoc_md1015", null ],
        [ "Quantum Circuit Model", "md__2home_2runner_2work_2numerics_2numerics_2apps_2quantum__demo_2README.html#autotoc_md1017", null ],
        [ "Gate Set", "md__2home_2runner_2work_2numerics_2numerics_2apps_2quantum__demo_2README.html#autotoc_md1019", [
          [ "Single-qubit gates", "md__2home_2runner_2work_2numerics_2numerics_2apps_2quantum__demo_2README.html#autotoc_md1020", null ],
          [ "Two-qubit gates", "md__2home_2runner_2work_2numerics_2numerics_2apps_2quantum__demo_2README.html#autotoc_md1021", null ],
          [ "Three-qubit gates", "md__2home_2runner_2work_2numerics_2numerics_2apps_2quantum__demo_2README.html#autotoc_md1022", null ]
        ] ],
        [ "Preset Circuits", "md__2home_2runner_2work_2numerics_2numerics_2apps_2quantum__demo_2README.html#autotoc_md1024", [
          [ "1 — Bell State $|\\Phi^+\\rangle$  (2 qubits)", "md__2home_2runner_2work_2numerics_2numerics_2apps_2quantum__demo_2README.html#autotoc_md1025", null ],
          [ "2 — GHZ State  (3 qubits)", "md__2home_2runner_2work_2numerics_2numerics_2apps_2quantum__demo_2README.html#autotoc_md1027", null ],
          [ "3 — Grover Search  (2 qubits, target $|11\\rangle$)", "md__2home_2runner_2work_2numerics_2numerics_2apps_2quantum__demo_2README.html#autotoc_md1029", null ],
          [ "4 — Quantum Teleportation  (3 qubits)", "md__2home_2runner_2work_2numerics_2numerics_2apps_2quantum__demo_2README.html#autotoc_md1031", null ],
          [ "5 — Quantum Fourier Transform  (3 qubits, input $|001\\rangle$)", "md__2home_2runner_2work_2numerics_2numerics_2apps_2quantum__demo_2README.html#autotoc_md1033", null ]
        ] ],
        [ "Simulation Model", "md__2home_2runner_2work_2numerics_2numerics_2apps_2quantum__demo_2README.html#autotoc_md1035", [
          [ "Statevector evolution", "md__2home_2runner_2work_2numerics_2numerics_2apps_2quantum__demo_2README.html#autotoc_md1036", null ],
          [ "Measurement sampling", "md__2home_2runner_2work_2numerics_2numerics_2apps_2quantum__demo_2README.html#autotoc_md1037", null ],
          [ "Observables", "md__2home_2runner_2work_2numerics_2numerics_2apps_2quantum__demo_2README.html#autotoc_md1038", null ]
        ] ],
        [ "Numerics Library Integration", "md__2home_2runner_2work_2numerics_2numerics_2apps_2quantum__demo_2README.html#autotoc_md1040", null ],
        [ "The <tt>num::Circuit</tt> API", "md__2home_2runner_2work_2numerics_2numerics_2apps_2quantum__demo_2README.html#autotoc_md1042", null ],
        [ "Visualisation", "md__2home_2runner_2work_2numerics_2numerics_2apps_2quantum__demo_2README.html#autotoc_md1044", null ],
        [ "Controls", "md__2home_2runner_2work_2numerics_2numerics_2apps_2quantum__demo_2README.html#autotoc_md1046", null ],
        [ "Build", "md__2home_2runner_2work_2numerics_2numerics_2apps_2quantum__demo_2README.html#autotoc_md1048", [
          [ "Using CMake presets (recommended)", "md__2home_2runner_2work_2numerics_2numerics_2apps_2quantum__demo_2README.html#autotoc_md1049", null ],
          [ "Manual configuration", "md__2home_2runner_2work_2numerics_2numerics_2apps_2quantum__demo_2README.html#autotoc_md1050", null ]
        ] ],
        [ "A Note on Quantum Error Correction", "md__2home_2runner_2work_2numerics_2numerics_2apps_2quantum__demo_2README.html#autotoc_md1052", null ],
        [ "References", "md__2home_2runner_2work_2numerics_2numerics_2apps_2quantum__demo_2README.html#autotoc_md1054", null ]
      ] ]
    ] ],
    [ "README", "md__2home_2runner_2work_2numerics_2numerics_2apps_2nbody_2README.html", [
      [ "Gravitational N-Body Simulation", "md__2home_2runner_2work_2numerics_2numerics_2apps_2nbody_2README.html#autotoc_md1055", [
        [ "Physics", "md__2home_2runner_2work_2numerics_2numerics_2apps_2nbody_2README.html#autotoc_md1057", null ],
        [ "Numerics Library Integration", "md__2home_2runner_2work_2numerics_2numerics_2apps_2nbody_2README.html#autotoc_md1059", [
          [ "Verlet integration (symplectic)", "md__2home_2runner_2work_2numerics_2numerics_2apps_2nbody_2README.html#autotoc_md1060", null ],
          [ "RK4 integration (non-symplectic)", "md__2home_2runner_2work_2numerics_2numerics_2apps_2nbody_2README.html#autotoc_md1061", null ]
        ] ],
        [ "Project layout", "md__2home_2runner_2work_2numerics_2numerics_2apps_2nbody_2README.html#autotoc_md1063", null ],
        [ "Running", "md__2home_2runner_2work_2numerics_2numerics_2apps_2nbody_2README.html#autotoc_md1064", null ],
        [ "References", "md__2home_2runner_2work_2numerics_2numerics_2apps_2nbody_2README.html#autotoc_md1065", null ]
      ] ]
    ] ],
    [ "Namespaces", "namespaces.html", [
      [ "Namespace List", "namespaces.html", "namespaces_dup" ],
      [ "Namespace Members", "namespacemembers.html", [
        [ "All", "namespacemembers.html", "namespacemembers_dup" ],
        [ "Functions", "namespacemembers_func.html", "namespacemembers_func" ],
        [ "Variables", "namespacemembers_vars.html", null ],
        [ "Typedefs", "namespacemembers_type.html", null ],
        [ "Enumerations", "namespacemembers_enum.html", null ]
      ] ]
    ] ],
    [ "Concepts", "concepts.html", "concepts" ],
    [ "Classes", "annotated.html", [
      [ "Class List", "annotated.html", "annotated_dup" ],
      [ "Class Index", "classes.html", null ],
      [ "Class Hierarchy", "hierarchy.html", "hierarchy" ],
      [ "Class Members", "functions.html", [
        [ "All", "functions.html", "functions_dup" ],
        [ "Functions", "functions_func.html", "functions_func" ],
        [ "Variables", "functions_vars.html", "functions_vars" ],
        [ "Typedefs", "functions_type.html", null ],
        [ "Enumerations", "functions_enum.html", null ]
      ] ]
    ] ],
    [ "Files", "files.html", [
      [ "File List", "files.html", "files_dup" ],
      [ "File Members", "globals.html", [
        [ "All", "globals.html", null ],
        [ "Functions", "globals_func.html", null ],
        [ "Variables", "globals_vars.html", null ],
        [ "Typedefs", "globals_type.html", null ]
      ] ]
    ] ]
  ] ]
];

var NAVTREEINDEX =
[
"adi_8hpp.html",
"classnum_1_1BandedMatrix.html#af1e93c1bca37ffae0ab8a998338bfb2e",
"classnum_1_1Matrix.html#ae1b894626f654b8e6b7cc2f71eb3b754",
"classtdse_1_1TDSESolver.html#a1faae7b66238d83ce73cc981405d57ea",
"fft_8hpp.html#ad4ea8f8efdab43a81b7b83366ac10f25",
"jacobi__eig_8hpp.html",
"namespacebackends_1_1seq.html",
"namespacenum.html#aff072a541903b1b76a64ddb928cbf981",
"ns__demo_2include_2ns__demo_2sim_8hpp_source.html",
"page_lu_notes.html#autotoc_md275",
"page_week1.html#autotoc_md585",
"pde_2stencil_8hpp.html#a892ab7e328eb43ed0933eb58d2efe57b",
"statevector_8cpp.html#a6eae7975ae635a360c4bf3677cca47f7",
"structnum_1_1EigenResult.html",
"structnum_1_1RK4__2ndSteps_1_1iterator.html#a91686df370f6d744c8615e078316cae2",
"structnum_1_1ode_1_1ImplicitParams.html#a8c251cfaf081207cf3c2e9e14f71157d",
"structphysics_1_1Particle3D.html#a95a3a02b94c933d28d54e6e21b1cdb6d"
];

var SYNCONMSG = 'click to disable panel synchronisation';
var SYNCOFFMSG = 'click to enable panel synchronisation';