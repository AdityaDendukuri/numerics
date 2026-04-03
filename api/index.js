var index =
[
    [ "Example: Lorenz attractor", "index.html#autotoc_md185", null ],
    [ "Example: 2D heat equation", "index.html#autotoc_md187", null ],
    [ "Library Modules", "index.html#autotoc_md189", [
      [ "core – Vectors, matrices, and backend dispatch", "index.html#autotoc_md190", null ],
      [ "factorization – Direct linear solvers", "index.html#autotoc_md191", null ],
      [ "solvers – Iterative solvers", "index.html#autotoc_md192", null ],
      [ "eigen – Eigenvalue methods", "index.html#autotoc_md193", null ],
      [ "svd – Singular value decomposition", "index.html#autotoc_md194", null ],
      [ "spectral – Fourier transforms", "index.html#autotoc_md195", null ],
      [ "analysis – Quadrature and root finding", "index.html#autotoc_md196", null ],
      [ "stats – Simulation observables", "index.html#autotoc_md197", null ],
      [ "markov – Markov chain Monte Carlo", "index.html#autotoc_md198", null ],
      [ "fields – Grid geometry and field storage", "index.html#autotoc_md199", null ],
      [ "solve – Unified solver dispatcher", "index.html#autotoc_md200", null ],
      [ "sparse and banded – Structured sparse formats", "index.html#autotoc_md201", null ],
      [ "meshfree – Particle-based spatial structures", "index.html#autotoc_md202", null ],
      [ "plot – Gnuplot pipe wrapper", "index.html#autotoc_md204", null ]
    ] ],
    [ "Physics Simulations", "index.html#autotoc_md206", [
      [ "Monte Carlo", "index.html#autotoc_md207", [
        [ "2D Ising Model", "index.html#autotoc_md208", null ]
      ] ],
      [ "Fluid Dynamics", "index.html#autotoc_md210", [
        [ "2D SPH Fluid Simulation", "index.html#autotoc_md211", null ],
        [ "3D SPH Fluid Simulation", "index.html#autotoc_md212", null ],
        [ "2D Incompressible Navier-Stokes", "index.html#autotoc_md213", null ]
      ] ],
      [ "Quantum Mechanics", "index.html#autotoc_md215", [
        [ "2D Time-Dependent Schrodinger Equation", "index.html#autotoc_md216", null ]
      ] ],
      [ "Electromagnetism", "index.html#autotoc_md218", [
        [ "Electromagnetic Field Demo", "index.html#autotoc_md219", null ]
      ] ]
    ] ],
    [ "Status Report", "index.html#autotoc_md221", null ],
    [ "Lecture Notes", "index.html#autotoc_md223", null ],
    [ "2D Heat Equation", "page_heat_demo.html", [
      [ "Code", "page_heat_demo.html#autotoc_md179", null ],
      [ "Figure", "page_heat_demo.html#autotoc_md181", null ],
      [ "Library features used", "page_heat_demo.html#autotoc_md183", null ]
    ] ],
    [ "High-Performance Computing", "page_performance.html", [
      [ "Performance Limits", "page_performance.html#sec_roofline", [
        [ "Why Clock Frequency Plateaued", "page_performance.html#autotoc_md375", null ],
        [ "Amdahl's Law", "page_performance.html#autotoc_md376", null ],
        [ "Roofline Model", "page_performance.html#autotoc_md377", null ],
        [ "Arithmetic Intensity of Common Kernels", "page_performance.html#autotoc_md378", null ],
        [ "Memory Hierarchy Latencies", "page_performance.html#autotoc_md379", null ]
      ] ],
      [ "Cache-Blocked Matrix Multiplication", "page_performance.html#sec_cache_blocking", [
        [ "Why Naive \\f$ijk\\f$ Matmul is Slow", "page_performance.html#autotoc_md381", null ],
        [ "Worked Cache-Miss Count for N = 512", "page_performance.html#autotoc_md382", null ],
        [ "The Tiling Idea", "page_performance.html#autotoc_md383", null ],
        [ "Outer Loop Order: \\f$ii \\to jj \\to kk\\f$", "page_performance.html#autotoc_md384", null ],
        [ "Inner Loop Order: \\f$i \\to k \\to j\\f$", "page_performance.html#autotoc_md385", null ],
        [ "Measured Speedup", "page_performance.html#autotoc_md386", null ]
      ] ],
      [ "Register Blocking", "page_performance.html#sec_register_blocking", [
        [ "The Residual Problem After Cache Blocking", "page_performance.html#autotoc_md388", null ],
        [ "The Register Tile", "page_performance.html#autotoc_md389", null ],
        [ "Why Scalar Register Blocking Regresses", "page_performance.html#autotoc_md390", null ],
        [ "Register Blocking + SIMD as a Unit", "page_performance.html#autotoc_md391", null ]
      ] ],
      [ "SIMD Vectorization", "page_performance.html#sec_simd", [
        [ "AVX-256 and ARM NEON", "page_performance.html#autotoc_md393", null ],
        [ "The SIMD Micro-kernel", "page_performance.html#autotoc_md394", null ],
        [ "Compile-Time Architecture Dispatch", "page_performance.html#autotoc_md395", null ],
        [ "Jacobi SVD SIMD Fusion", "page_performance.html#autotoc_md396", null ],
        [ "Measured Progression", "page_performance.html#autotoc_md397", null ],
        [ "matvec_simd: Bandwidth Bound", "page_performance.html#autotoc_md398", null ]
      ] ]
    ] ],
    [ "Parallel and Distributed Computing", "page_parallel.html", [
      [ "MPI: Distributed Memory", "page_parallel.html#sec_mpi", [
        [ "The SPMD Model", "page_parallel.html#autotoc_md326", null ],
        [ "Point-to-Point Communication", "page_parallel.html#autotoc_md327", null ],
        [ "Collective Operations", "page_parallel.html#autotoc_md328", null ],
        [ "Distributed Dot Product", "page_parallel.html#autotoc_md329", null ],
        [ "Distributed Matrix-Vector Product", "page_parallel.html#autotoc_md330", null ],
        [ "Parallel Conjugate Gradient", "page_parallel.html#autotoc_md331", null ],
        [ "Hiding Latency with Non-Blocking Collectives", "page_parallel.html#autotoc_md332", null ]
      ] ],
      [ "CUDA: GPU Computing", "page_parallel.html#sec_cuda", [
        [ "GPU Hierarchy", "page_parallel.html#autotoc_md334", null ],
        [ "GPU Memory Hierarchy", "page_parallel.html#autotoc_md335", null ],
        [ "Coalesced Global Memory Access", "page_parallel.html#autotoc_md336", null ],
        [ "Tiled Matrix Multiplication with Shared Memory", "page_parallel.html#autotoc_md337", null ],
        [ "Kernel Launch Overhead and Crossover Points", "page_parallel.html#autotoc_md338", null ],
        [ "Batched Thomas Algorithm", "page_parallel.html#autotoc_md339", null ]
      ] ],
      [ "Banded Matrix Systems", "page_parallel.html#sec_banded", [
        [ "Band Structure", "page_parallel.html#autotoc_md341", null ],
        [ "LAPACK-Style Band Storage", "page_parallel.html#autotoc_md342", null ],
        [ "Banded LU with Partial Pivoting", "page_parallel.html#autotoc_md343", null ],
        [ "Numerical Stability: Growth Factor", "page_parallel.html#autotoc_md344", null ],
        [ "Solve Phase", "page_parallel.html#autotoc_md345", null ]
      ] ]
    ] ],
    [ "Dense Matrix Factorizations", "page_factorizations.html", [
      [ "Why Factorizations?", "page_factorizations.html#sec_factorization_motivation", null ],
      [ "LU Factorization with Partial Pivoting", "page_factorizations.html#sec_lu", [
        [ "The PA = LU Form", "page_factorizations.html#autotoc_md117", null ],
        [ "Doolittle Algorithm", "page_factorizations.html#autotoc_md118", null ],
        [ "Why Partial Pivoting?", "page_factorizations.html#autotoc_md119", null ],
        [ "Solving with the Factorization", "page_factorizations.html#autotoc_md120", null ],
        [ "Determinant and Inverse", "page_factorizations.html#autotoc_md121", null ],
        [ "Complexity and Stability", "page_factorizations.html#autotoc_md122", null ],
        [ "API", "page_factorizations.html#autotoc_md123", null ]
      ] ],
      [ "QR Factorization (Householder)", "page_factorizations.html#sec_qr", [
        [ "The A = QR Form", "page_factorizations.html#autotoc_md125", null ],
        [ "Householder Reflectors", "page_factorizations.html#autotoc_md126", null ],
        [ "Algorithm", "page_factorizations.html#autotoc_md127", null ],
        [ "Why Householder Beats Gram-Schmidt", "page_factorizations.html#autotoc_md128", null ],
        [ "Least-Squares Solve", "page_factorizations.html#autotoc_md129", null ],
        [ "Complexity", "page_factorizations.html#autotoc_md130", null ],
        [ "API", "page_factorizations.html#autotoc_md131", null ]
      ] ]
    ] ],
    [ "Linear Solvers", "page_linear_solvers.html", [
      [ "Thomas Algorithm (Tridiagonal)", "page_linear_solvers.html#sec_thomas", [
        [ "The Tridiagonal System", "page_linear_solvers.html#autotoc_md242", null ],
        [ "LU Factorization View", "page_linear_solvers.html#autotoc_md243", null ],
        [ "Forward Sweep and Back Substitution", "page_linear_solvers.html#autotoc_md244", null ],
        [ "Complexity and Stability", "page_linear_solvers.html#autotoc_md245", null ],
        [ "GPU Batched Variant", "page_linear_solvers.html#autotoc_md246", null ],
        [ "API", "page_linear_solvers.html#autotoc_md247", null ]
      ] ],
      [ "Stationary Iterative Methods", "page_linear_solvers.html#sec_stationary", [
        [ "Splitting Framework", "page_linear_solvers.html#autotoc_md249", null ],
        [ "Jacobi Iteration", "page_linear_solvers.html#autotoc_md250", null ],
        [ "Gauss-Seidel and SOR", "page_linear_solvers.html#autotoc_md251", null ]
      ] ],
      [ "Conjugate Gradient", "page_linear_solvers.html#sec_cg", [
        [ "Minimization Perspective", "page_linear_solvers.html#autotoc_md253", null ],
        [ "A-Conjugacy", "page_linear_solvers.html#autotoc_md254", null ],
        [ "Step Length and Direction Update", "page_linear_solvers.html#autotoc_md255", null ],
        [ "Convergence", "page_linear_solvers.html#autotoc_md256", null ],
        [ "Preconditioned CG (PCG)", "page_linear_solvers.html#autotoc_md257", null ],
        [ "Matrix-Free CG", "page_linear_solvers.html#autotoc_md258", null ],
        [ "API", "page_linear_solvers.html#autotoc_md259", null ]
      ] ],
      [ "GMRES", "page_linear_solvers.html#sec_gmres", [
        [ "Problem and Approach", "page_linear_solvers.html#autotoc_md261", null ],
        [ "Arnoldi Relation", "page_linear_solvers.html#autotoc_md262", null ],
        [ "The GMRES Least-Squares Problem", "page_linear_solvers.html#autotoc_md263", null ],
        [ "Restart: GMRES(\\f$m\\f$)", "page_linear_solvers.html#autotoc_md264", null ],
        [ "Convergence", "page_linear_solvers.html#autotoc_md265", null ],
        [ "API", "page_linear_solvers.html#autotoc_md266", null ]
      ] ]
    ] ],
    [ "Eigenvalue Algorithms", "page_eigenvalues.html", [
      [ "Cyclic Jacobi Eigendecomposition", "page_eigenvalues.html#sec_jacobi_eig", [
        [ "Similarity Transform", "page_eigenvalues.html#autotoc_md98", null ],
        [ "Rotation Parameters", "page_eigenvalues.html#autotoc_md99", null ],
        [ "Diagonal and Off-Diagonal Update Formulas", "page_eigenvalues.html#autotoc_md100", null ],
        [ "Convergence", "page_eigenvalues.html#autotoc_md101", null ],
        [ "Complexity", "page_eigenvalues.html#autotoc_md102", null ],
        [ "Comparison with Other Eigensolvers", "page_eigenvalues.html#autotoc_md103", null ]
      ] ],
      [ "Power, Inverse, and Rayleigh Iteration", "page_eigenvalues.html#sec_power", [
        [ "Power Iteration", "page_eigenvalues.html#autotoc_md105", null ],
        [ "Inverse Iteration", "page_eigenvalues.html#autotoc_md106", null ],
        [ "Rayleigh Quotient Iteration", "page_eigenvalues.html#autotoc_md107", null ],
        [ "Method Selection", "page_eigenvalues.html#autotoc_md108", null ]
      ] ],
      [ "Lanczos Algorithm", "page_eigenvalues.html#sec_lanczos", [
        [ "The Lanczos Relation", "page_eigenvalues.html#autotoc_md110", null ],
        [ "Ritz Values and Cheap Residual Bound", "page_eigenvalues.html#autotoc_md111", null ],
        [ "Ghost Eigenvalues and Full Reorthogonalization", "page_eigenvalues.html#autotoc_md112", null ],
        [ "Thick Restart and ARPACK", "page_eigenvalues.html#autotoc_md113", null ],
        [ "Complexity", "page_eigenvalues.html#autotoc_md114", null ]
      ] ]
    ] ],
    [ "Singular Value Decomposition", "page_svd.html", [
      [ "One-Sided Jacobi SVD", "page_svd.html#sec_svd_jacobi", [
        [ "Why Not Form \\f$A^T A\\f$", "page_svd.html#autotoc_md556", null ],
        [ "Column Orthogonality Condition", "page_svd.html#autotoc_md557", null ],
        [ "Rotation Angle", "page_svd.html#autotoc_md558", null ],
        [ "Convergence", "page_svd.html#autotoc_md559", null ],
        [ "High Relative Accuracy", "page_svd.html#autotoc_md560", null ],
        [ "Complexity and Comparison", "page_svd.html#autotoc_md561", null ]
      ] ],
      [ "Randomized Truncated SVD", "page_svd.html#sec_svd_randomized", [
        [ "Eckart-Young Theorem", "page_svd.html#autotoc_md563", null ],
        [ "Algorithm (Halko-Martinsson-Tropp 2011)", "page_svd.html#autotoc_md564", null ],
        [ "HMT Error Bound", "page_svd.html#autotoc_md565", null ],
        [ "Oversampling Cap \\f$\\ell \\leq \\min(m,n)\\f$", "page_svd.html#autotoc_md566", null ],
        [ "Accuracy vs Cost Trade-offs", "page_svd.html#autotoc_md567", null ]
      ] ]
    ] ],
    [ "Fast Fourier Transform", "page_fft.html", [
      [ "The Discrete Fourier Transform", "page_fft.html#sec_dft", [
        [ "Interpretation of frequency bins", "page_fft.html#autotoc_md133", null ],
        [ "Parseval's theorem", "page_fft.html#autotoc_md134", null ]
      ] ],
      [ "Naive DFT: O(n^2) cost", "page_fft.html#sec_naive", null ],
      [ "Cooley-Tukey Radix-2 DIT", "page_fft.html#sec_cooley_tukey", [
        [ "Recurrence and complexity", "page_fft.html#autotoc_md137", null ],
        [ "Bit-reversal permutation", "page_fft.html#autotoc_md138", null ]
      ] ],
      [ "Algorithm (iterative DIT)", "page_fft.html#sec_algorithm", null ],
      [ "Real-to-Complex Transform (rfft)", "page_fft.html#sec_rfft", null ],
      [ "FFTPlan: amortizing planning cost", "page_fft.html#sec_fftplan", null ],
      [ "Backends", "page_fft.html#sec_backends", [
        [ "seq backend", "page_fft.html#autotoc_md143", null ],
        [ "fftw backend", "page_fft.html#autotoc_md144", null ]
      ] ],
      [ "API Reference", "page_fft.html#sec_api", null ],
      [ "Worked Example", "page_fft.html#sec_example", null ]
    ] ],
    [ "Numerical Analysis", "page_analysis.html", [
      [ "Numerical Integration (Quadrature)", "page_analysis.html#sec_quadrature", [
        [ "Newton-Cotes Rules", "page_analysis.html#autotoc_md3", [
          [ "Trapezoidal Rule", "page_analysis.html#autotoc_md4", null ],
          [ "Simpson's 1/3 Rule", "page_analysis.html#autotoc_md5", null ]
        ] ],
        [ "Gauss-Legendre Quadrature", "page_analysis.html#autotoc_md6", null ],
        [ "Adaptive Simpson", "page_analysis.html#autotoc_md7", null ],
        [ "Romberg Integration", "page_analysis.html#autotoc_md8", null ],
        [ "Method Comparison", "page_analysis.html#autotoc_md9", null ]
      ] ],
      [ "Root Finding", "page_analysis.html#sec_roots", [
        [ "Bisection", "page_analysis.html#autotoc_md11", null ],
        [ "Newton-Raphson", "page_analysis.html#autotoc_md12", null ],
        [ "Secant Method", "page_analysis.html#autotoc_md13", null ],
        [ "Brent's Method", "page_analysis.html#autotoc_md14", null ],
        [ "Method Comparison", "page_analysis.html#autotoc_md15", null ]
      ] ]
    ] ],
    [ "Week 1: Parallel Computing Fundamentals & Linear Algebra", "page_week1.html", [
      [ "1. Why Parallel Computing?", "page_week1.html#autotoc_md568", null ],
      [ "2. Taxonomy of Parallel Architectures", "page_week1.html#autotoc_md570", [
        [ "Flynn's Taxonomy", "page_week1.html#autotoc_md571", null ],
        [ "Memory Models", "page_week1.html#autotoc_md572", null ]
      ] ],
      [ "3. Performance Limits", "page_week1.html#autotoc_md574", [
        [ "Amdahl's Law", "page_week1.html#autotoc_md575", null ],
        [ "Gustafson's Law", "page_week1.html#autotoc_md576", null ],
        [ "Roofline Model", "page_week1.html#autotoc_md577", null ]
      ] ],
      [ "4. Linear Algebra Fundamentals", "page_week1.html#autotoc_md579", [
        [ "Vectors", "page_week1.html#autotoc_md580", null ],
        [ "Matrices", "page_week1.html#autotoc_md581", null ],
        [ "Matrix-Vector Multiplication", "page_week1.html#autotoc_md582", null ],
        [ "Matrix-Matrix Multiplication", "page_week1.html#autotoc_md583", null ]
      ] ],
      [ "5. Parallelizing Linear Algebra", "page_week1.html#autotoc_md585", [
        [ "Data Decomposition", "page_week1.html#autotoc_md586", null ],
        [ "Parallel Dot Product", "page_week1.html#autotoc_md587", null ],
        [ "Parallel Matrix-Vector Product", "page_week1.html#autotoc_md588", null ],
        [ "Parallel Matrix-Matrix Product", "page_week1.html#autotoc_md589", null ]
      ] ],
      [ "6. Key Takeaways", "page_week1.html#autotoc_md591", null ],
      [ "Exercises", "page_week1.html#autotoc_md593", null ],
      [ "References", "page_week1.html#autotoc_md595", null ]
    ] ],
    [ "Week 2: MPI -- Distributed Memory Programming", "page_week2.html", [
      [ "1. The Message Passing Model", "page_week2.html#autotoc_md596", null ],
      [ "2. MPI Basics", "page_week2.html#autotoc_md598", [
        [ "Initialization and Finalization", "page_week2.html#autotoc_md599", null ],
        [ "Communicators", "page_week2.html#autotoc_md600", null ]
      ] ],
      [ "3. Point-to-Point Communication", "page_week2.html#autotoc_md602", [
        [ "Blocking Send/Receive", "page_week2.html#autotoc_md603", null ],
        [ "Common MPI Datatypes", "page_week2.html#autotoc_md604", null ],
        [ "Deadlock", "page_week2.html#autotoc_md605", null ],
        [ "Non-Blocking Communication", "page_week2.html#autotoc_md606", null ]
      ] ],
      [ "4. Collective Operations", "page_week2.html#autotoc_md608", [
        [ "Broadcast", "page_week2.html#autotoc_md609", null ],
        [ "Reduce", "page_week2.html#autotoc_md610", null ],
        [ "Allreduce", "page_week2.html#autotoc_md611", null ],
        [ "Gather and Scatter", "page_week2.html#autotoc_md612", null ],
        [ "Allgather", "page_week2.html#autotoc_md613", null ],
        [ "Summary Table", "page_week2.html#autotoc_md614", null ]
      ] ],
      [ "5. Distributed Linear Algebra with MPI", "page_week2.html#autotoc_md616", [
        [ "Distributed Vector", "page_week2.html#autotoc_md617", null ],
        [ "Distributed Dot Product", "page_week2.html#autotoc_md618", null ],
        [ "Distributed Norm", "page_week2.html#autotoc_md619", null ],
        [ "Distributed Matrix-Vector Product", "page_week2.html#autotoc_md620", null ]
      ] ],
      [ "6. Performance Considerations", "page_week2.html#autotoc_md622", [
        [ "Communication vs. Computation", "page_week2.html#autotoc_md623", null ],
        [ "Hiding Latency", "page_week2.html#autotoc_md624", null ],
        [ "Load Balancing", "page_week2.html#autotoc_md625", null ]
      ] ],
      [ "7. Example: Parallel Conjugate Gradient", "page_week2.html#autotoc_md627", null ],
      [ "8. Our Library's MPI Interface", "page_week2.html#autotoc_md629", null ],
      [ "Exercises", "page_week2.html#autotoc_md631", null ],
      [ "References", "page_week2.html#autotoc_md633", null ]
    ] ],
    [ "Week 3: CUDA -- GPU Programming", "page_week3.html", [
      [ "1. GPU Architecture", "page_week3.html#autotoc_md634", [
        [ "CPU vs. GPU Philosophy", "page_week3.html#autotoc_md635", null ],
        [ "NVIDIA GPU Hierarchy", "page_week3.html#autotoc_md636", null ]
      ] ],
      [ "2. CUDA Programming Model", "page_week3.html#autotoc_md638", [
        [ "Kernels and Threads", "page_week3.html#autotoc_md639", null ],
        [ "Thread Hierarchy", "page_week3.html#autotoc_md640", null ],
        [ "Launch Configuration", "page_week3.html#autotoc_md641", null ]
      ] ],
      [ "3. Memory Hierarchy", "page_week3.html#autotoc_md643", [
        [ "Memory Types", "page_week3.html#autotoc_md644", null ],
        [ "Global Memory", "page_week3.html#autotoc_md645", null ],
        [ "Shared Memory", "page_week3.html#autotoc_md646", null ]
      ] ],
      [ "4. Common Patterns", "page_week3.html#autotoc_md648", [
        [ "Map (Element-wise Operations)", "page_week3.html#autotoc_md649", null ],
        [ "Reduction", "page_week3.html#autotoc_md650", null ],
        [ "Stencil (Neighbors)", "page_week3.html#autotoc_md651", null ]
      ] ],
      [ "5. Matrix Operations on GPU", "page_week3.html#autotoc_md653", [
        [ "Matrix-Vector Multiplication", "page_week3.html#autotoc_md654", null ],
        [ "Matrix-Matrix Multiplication", "page_week3.html#autotoc_md655", null ]
      ] ],
      [ "6. Performance Optimization", "page_week3.html#autotoc_md657", [
        [ "Coalesced Memory Access", "page_week3.html#autotoc_md658", null ],
        [ "Occupancy", "page_week3.html#autotoc_md659", null ],
        [ "Avoiding Warp Divergence", "page_week3.html#autotoc_md660", null ],
        [ "Use the Profiler", "page_week3.html#autotoc_md661", null ]
      ] ],
      [ "7. Our Library's CUDA Interface", "page_week3.html#autotoc_md663", null ],
      [ "8. CPU vs GPU: When to Use What", "page_week3.html#autotoc_md665", null ],
      [ "Exercises", "page_week3.html#autotoc_md667", null ],
      [ "References", "page_week3.html#autotoc_md669", null ]
    ] ],
    [ "Week 4: Linear Solvers -- Theory and Parallel Implementation", "page_week4.html", [
      [ "1. Introduction", "page_week4.html#autotoc_md670", null ],
      [ "2. Conjugate Gradient Method", "page_week4.html#autotoc_md672", [
        [ "2.1 Mathematical Foundation", "page_week4.html#autotoc_md673", [
          [ "The Minimization Perspective", "page_week4.html#autotoc_md674", null ],
          [ "A-Conjugacy and Optimality", "page_week4.html#autotoc_md675", null ],
          [ "Krylov Subspace Connection", "page_week4.html#autotoc_md676", null ]
        ] ],
        [ "2.2 Algorithm", "page_week4.html#autotoc_md677", null ],
        [ "2.3 The CG Algorithm", "page_week4.html#autotoc_md678", null ],
        [ "2.4 Convergence Analysis", "page_week4.html#autotoc_md679", null ],
        [ "2.5 Operation Count", "page_week4.html#autotoc_md680", null ]
      ] ],
      [ "3. Thomas Algorithm (Tridiagonal Solver)", "page_week4.html#autotoc_md682", [
        [ "3.1 Problem Structure", "page_week4.html#autotoc_md683", null ],
        [ "3.2 Thomas Algorithm via LU Structure", "page_week4.html#autotoc_md684", null ],
        [ "3.3 The Thomas Algorithm", "page_week4.html#autotoc_md685", null ],
        [ "3.4 Complexity Analysis", "page_week4.html#autotoc_md686", null ],
        [ "3.5 Stability Analysis", "page_week4.html#autotoc_md687", null ]
      ] ],
      [ "4. CUDA Implementation", "page_week4.html#autotoc_md689", [
        [ "4.1 CG on GPU", "page_week4.html#autotoc_md690", [
          [ "GPU Kernels Used", "page_week4.html#autotoc_md691", null ],
          [ "Performance Considerations", "page_week4.html#autotoc_md692", null ]
        ] ],
        [ "4.2 Thomas Algorithm on GPU", "page_week4.html#autotoc_md693", [
          [ "Batched Thomas Kernel", "page_week4.html#autotoc_md694", null ],
          [ "When Batched Thomas Helps", "page_week4.html#autotoc_md695", null ]
        ] ]
      ] ],
      [ "5. MPI Implementation (Distributed Memory)", "page_week4.html#autotoc_md697", [
        [ "5.1 Distributed CG", "page_week4.html#autotoc_md698", [
          [ "Data Distribution", "page_week4.html#autotoc_md699", null ],
          [ "Distributed Operations", "page_week4.html#autotoc_md700", null ],
          [ "Distributed CG Algorithm", "page_week4.html#autotoc_md701", null ],
          [ "Communication Analysis", "page_week4.html#autotoc_md702", null ]
        ] ],
        [ "5.2 Distributed Thomas (Pipeline)", "page_week4.html#autotoc_md703", null ]
      ] ],
      [ "6. Hybrid MPI+CUDA Implementation", "page_week4.html#autotoc_md705", null ],
      [ "7. Summary", "page_week4.html#autotoc_md707", null ]
    ] ],
    [ "Week 5: Banded Matrix Solvers -- Theory and HPC Implementation", "page_week5.html", [
      [ "1. Introduction", "page_week5.html#autotoc_md708", null ],
      [ "2. Band Storage Formats", "page_week5.html#autotoc_md710", [
        [ "2.1 LAPACK Band Storage (Column-Major)", "page_week5.html#autotoc_md711", null ],
        [ "2.2 Memory Layout Benefits", "page_week5.html#autotoc_md712", null ]
      ] ],
      [ "3. LU Factorization for Banded Matrices", "page_week5.html#autotoc_md714", [
        [ "3.1 Mathematical Foundation", "page_week5.html#autotoc_md715", null ],
        [ "3.2 Algorithm", "page_week5.html#autotoc_md716", null ],
        [ "3.3 The Algorithm", "page_week5.html#autotoc_md717", null ],
        [ "3.4 Solving After Factorization", "page_week5.html#autotoc_md718", null ]
      ] ],
      [ "4. Complexity Analysis", "page_week5.html#autotoc_md720", [
        [ "4.1 Operation Counts", "page_week5.html#autotoc_md721", null ],
        [ "4.2 Comparison with Dense and Tridiagonal", "page_week5.html#autotoc_md722", null ],
        [ "4.3 Memory Bandwidth Analysis", "page_week5.html#autotoc_md723", null ]
      ] ],
      [ "5. Numerical Stability", "page_week5.html#autotoc_md725", [
        [ "5.1 Partial Pivoting Guarantees", "page_week5.html#autotoc_md726", null ],
        [ "5.2 Condition Number and Error", "page_week5.html#autotoc_md727", null ],
        [ "5.3 Diagonal Dominance", "page_week5.html#autotoc_md728", null ]
      ] ],
      [ "6. Implementation Details", "page_week5.html#autotoc_md730", [
        [ "6.1 BandedMatrix Class Design", "page_week5.html#autotoc_md731", null ],
        [ "6.2 SIMD Optimization", "page_week5.html#autotoc_md732", null ],
        [ "6.3 Cache Optimization", "page_week5.html#autotoc_md733", null ],
        [ "6.4 Multiple Right-Hand Sides", "page_week5.html#autotoc_md734", null ]
      ] ],
      [ "7. GPU Implementation", "page_week5.html#autotoc_md736", [
        [ "7.1 Batched Banded Solver", "page_week5.html#autotoc_md737", null ],
        [ "7.2 When GPU is Beneficial", "page_week5.html#autotoc_md738", null ]
      ] ],
      [ "8. Application: Radiative Transfer", "page_week5.html#autotoc_md740", [
        [ "8.1 Two-Stream Approximation", "page_week5.html#autotoc_md741", null ],
        [ "8.2 TUVX Photolysis Rates", "page_week5.html#autotoc_md742", null ],
        [ "8.3 Performance Considerations for NCAR Derecho", "page_week5.html#autotoc_md743", null ]
      ] ],
      [ "9. Summary and Best Practices", "page_week5.html#autotoc_md745", [
        [ "9.1 Algorithm Selection Guide", "page_week5.html#autotoc_md746", null ],
        [ "9.2 Numerical Robustness Checklist", "page_week5.html#autotoc_md747", null ],
        [ "9.3 Performance Optimization Checklist", "page_week5.html#autotoc_md748", null ],
        [ "9.4 Code References", "page_week5.html#autotoc_md749", null ]
      ] ],
      [ "Appendix A: Fill-in Bound", "page_week5.html#autotoc_md751", null ],
      [ "Appendix B: Diagonal Dominance Preserved Under Elimination", "page_week5.html#autotoc_md753", null ]
    ] ],
    [ "Week 6: Cache-Aware Computing -- Blocked Matrix Multiplication", "page_week6.html", [
      [ "Overview", "page_week6.html#autotoc_md754", null ],
      [ "1. The Memory Hierarchy", "page_week6.html#autotoc_md756", null ],
      [ "2. Arithmetic Intensity and the Roofline Model", "page_week6.html#autotoc_md758", null ],
      [ "3. Why Naive Matrix Multiply is Slow", "page_week6.html#autotoc_md760", [
        [ "Code (<tt>src/core/matrix.cpp:87-97</tt>)", "page_week6.html#autotoc_md761", null ]
      ] ],
      [ "4. The Blocking Idea", "page_week6.html#autotoc_md763", null ],
      [ "5. The Algorithm", "page_week6.html#autotoc_md765", [
        [ "Outer loop structure (<tt>src/core/matrix.cpp:153-161</tt>)", "page_week6.html#autotoc_md766", null ],
        [ "Micro-kernel (<tt>src/core/matrix.cpp:163-170</tt>)", "page_week6.html#autotoc_md767", null ]
      ] ],
      [ "6. Implementation Walkthrough", "page_week6.html#autotoc_md769", [
        [ "Declaration (<tt>include/core/matrix.hpp:48-64</tt>)", "page_week6.html#autotoc_md770", null ],
        [ "Full implementation (<tt>src/core/matrix.cpp:148-172</tt>)", "page_week6.html#autotoc_md771", null ]
      ] ],
      [ "7. Benchmark Results", "page_week6.html#autotoc_md773", null ],
      [ "8. Worked Cache-Miss Count", "page_week6.html#autotoc_md775", [
        [ "Naive loop – B accesses", "page_week6.html#autotoc_md776", null ],
        [ "Blocked loop – B accesses", "page_week6.html#autotoc_md777", null ]
      ] ],
      [ "9. Choosing the Block Size", "page_week6.html#autotoc_md779", null ],
      [ "10. What Comes Next", "page_week6.html#autotoc_md781", [
        [ "Step 1 – Register blocking", "page_week6.html#autotoc_md782", null ],
        [ "Step 2 – Explicit SIMD", "page_week6.html#autotoc_md783", null ]
      ] ],
      [ "11. Key Takeaways", "page_week6.html#autotoc_md785", null ],
      [ "Exercises", "page_week6.html#autotoc_md787", null ],
      [ "References", "page_week6.html#autotoc_md789", null ]
    ] ],
    [ "Week 7: Register Blocking -- What It Is and Why It Needs SIMD", "page_week7.html", [
      [ "Overview", "page_week7.html#autotoc_md790", null ],
      [ "1. The Residual Bottleneck After Cache Blocking", "page_week7.html#autotoc_md792", null ],
      [ "2. Register Blocking: The Idea", "page_week7.html#autotoc_md794", null ],
      [ "3. Implementation", "page_week7.html#autotoc_md796", [
        [ "Declaration (<tt>include/core/matrix.hpp:63-77</tt>)", "page_week7.html#autotoc_md797", null ],
        [ "Full implementation (<tt>src/core/matrix.cpp</tt>)", "page_week7.html#autotoc_md798", null ]
      ] ],
      [ "4. Benchmark Results – A Surprise", "page_week7.html#autotoc_md800", null ],
      [ "5. Why It Didn't Help: The Vectorisation Problem", "page_week7.html#autotoc_md802", [
        [ "What matmul_blocked does well", "page_week7.html#autotoc_md803", null ],
        [ "What register blocking does", "page_week7.html#autotoc_md804", null ],
        [ "The hardware angle", "page_week7.html#autotoc_md805", null ]
      ] ],
      [ "6. The Correct Mental Model: Register Blocking + SIMD Together", "page_week7.html#autotoc_md807", null ],
      [ "7. The BLAS Micro-kernel Design", "page_week7.html#autotoc_md809", null ],
      [ "8. What We Learned", "page_week7.html#autotoc_md811", null ],
      [ "9. Progression and Next Step", "page_week7.html#autotoc_md813", null ],
      [ "10. Key Takeaways", "page_week7.html#autotoc_md815", null ],
      [ "Exercises", "page_week7.html#autotoc_md817", null ],
      [ "References", "page_week7.html#autotoc_md819", null ]
    ] ],
    [ "Week 8: Explicit SIMD -- AVX-256 and ARM NEON", "page_week8.html", [
      [ "Overview", "page_week8.html#autotoc_md820", null ],
      [ "1. SIMD Fundamentals", "page_week8.html#autotoc_md822", null ],
      [ "2. The Micro-kernel", "page_week8.html#autotoc_md824", [
        [ "Why register blocking + SIMD work together", "page_week8.html#autotoc_md825", null ],
        [ "AVX-256 tile (<tt>src/core/matrix_simd.cpp</tt>)", "page_week8.html#autotoc_md826", null ],
        [ "ARM NEON tile (<tt>src/core/matrix_simd.cpp</tt>)", "page_week8.html#autotoc_md827", null ]
      ] ],
      [ "3. matvec_simd: Dot Product with Horizontal Reduction", "page_week8.html#autotoc_md829", [
        [ "AVX-256 version", "page_week8.html#autotoc_md830", null ],
        [ "NEON version", "page_week8.html#autotoc_md831", null ],
        [ "matvec benchmark results", "page_week8.html#autotoc_md832", null ]
      ] ],
      [ "4. Compile-Time Dispatch", "page_week8.html#autotoc_md834", [
        [ "Architecture detection (<tt>CMakeLists.txt</tt>)", "page_week8.html#autotoc_md835", null ],
        [ "Dispatch function (<tt>src/core/matrix_simd.cpp</tt>)", "page_week8.html#autotoc_md836", null ]
      ] ],
      [ "5. Boundary Handling", "page_week8.html#autotoc_md838", null ],
      [ "6. Complete Benchmark Results", "page_week8.html#autotoc_md840", [
        [ "Progression table", "page_week8.html#autotoc_md841", null ]
      ] ],
      [ "7. Key Takeaways", "page_week8.html#autotoc_md843", null ],
      [ "8. Exercises", "page_week8.html#autotoc_md845", null ],
      [ "References", "page_week8.html#autotoc_md847", null ]
    ] ],
    [ "Week 9: Dense Matrix Factorizations -- LU and QR", "page_week9.html", [
      [ "Overview", "page_week9.html#autotoc_md848", null ],
      [ "1. Why Factorizations?", "page_week9.html#autotoc_md850", null ],
      [ "2. LU Factorization with Partial Pivoting", "page_week9.html#autotoc_md852", [
        [ "The idea: Gaussian elimination with bookkeeping", "page_week9.html#autotoc_md853", null ],
        [ "Why partial pivoting?", "page_week9.html#autotoc_md854", null ],
        [ "Storage: packed L and U", "page_week9.html#autotoc_md855", null ],
        [ "Solving with LU (<tt>src/factorization/lu.cpp:54-83</tt>)", "page_week9.html#autotoc_md856", null ],
        [ "Determinant and inverse", "page_week9.html#autotoc_md857", null ]
      ] ],
      [ "3. QR Factorization via Householder Reflections", "page_week9.html#autotoc_md859", [
        [ "The idea", "page_week9.html#autotoc_md860", null ],
        [ "Why Householder instead of Gram-Schmidt?", "page_week9.html#autotoc_md861", null ],
        [ "The Householder reflector", "page_week9.html#autotoc_md862", null ],
        [ "Algorithm (<tt>src/factorization/qr.cpp:65-103</tt>)", "page_week9.html#autotoc_md863", null ],
        [ "Building Q explicitly", "page_week9.html#autotoc_md864", null ],
        [ "Least-squares solve (<tt>src/factorization/qr.cpp:119-136</tt>)", "page_week9.html#autotoc_md865", null ],
        [ "Sign of R's diagonal", "page_week9.html#autotoc_md866", null ]
      ] ],
      [ "4. API Reference", "page_week9.html#autotoc_md868", [
        [ "Declarations (<tt>include/factorization/</tt>)", "page_week9.html#autotoc_md869", null ],
        [ "Usage example", "page_week9.html#autotoc_md870", null ]
      ] ],
      [ "5. What These Unlock", "page_week9.html#autotoc_md872", null ],
      [ "6. Complexity and Numerical Properties", "page_week9.html#autotoc_md874", null ],
      [ "7. Key Takeaways", "page_week9.html#autotoc_md876", null ],
      [ "Exercises", "page_week9.html#autotoc_md878", null ],
      [ "References", "page_week9.html#autotoc_md880", null ]
    ] ]
];