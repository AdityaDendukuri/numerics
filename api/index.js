var index =
[
    [ "Example: Lorenz attractor", "index.html#autotoc_md56", null ],
    [ "Example: 2D heat equation", "index.html#autotoc_md58", null ],
    [ "Library Modules", "index.html#autotoc_md60", [
      [ "kernel – Performance substrate", "index.html#autotoc_md61", null ],
      [ "core – Vectors, matrices, and high-level dispatch", "index.html#autotoc_md62", null ],
      [ "factorization – Direct linear solvers", "index.html#autotoc_md63", null ],
      [ "solvers – Iterative solvers", "index.html#autotoc_md64", null ],
      [ "eigen – Eigenvalue methods", "index.html#autotoc_md65", null ],
      [ "svd – Singular value decomposition", "index.html#autotoc_md66", null ],
      [ "spectral – Fourier transforms", "index.html#autotoc_md67", null ],
      [ "analysis – Quadrature and root finding", "index.html#autotoc_md68", null ],
      [ "stats – Simulation observables", "index.html#autotoc_md69", null ],
      [ "markov – Markov chain Monte Carlo", "index.html#autotoc_md70", null ],
      [ "fields – Grid geometry and field storage", "index.html#autotoc_md71", null ],
      [ "solve – Unified solver dispatcher", "index.html#autotoc_md72", null ],
      [ "sparse and banded – Structured sparse formats", "index.html#autotoc_md73", null ],
      [ "meshfree – Particle-based spatial structures", "index.html#autotoc_md74", null ],
      [ "plot – Gnuplot pipe wrapper", "index.html#autotoc_md76", null ]
    ] ],
    [ "Physics Simulations", "index.html#autotoc_md78", [
      [ "Monte Carlo", "index.html#autotoc_md79", [
        [ "2D Ising Model", "index.html#autotoc_md80", null ]
      ] ],
      [ "Fluid Dynamics", "index.html#autotoc_md82", [
        [ "2D SPH Fluid Simulation", "index.html#autotoc_md83", null ],
        [ "3D SPH Fluid Simulation", "index.html#autotoc_md84", null ],
        [ "2D Incompressible Navier-Stokes", "index.html#autotoc_md85", null ]
      ] ],
      [ "Quantum Mechanics", "index.html#autotoc_md87", [
        [ "2D Time-Dependent Schrodinger Equation", "index.html#autotoc_md88", null ]
      ] ],
      [ "Electromagnetism", "index.html#autotoc_md90", [
        [ "Electromagnetic Field Demo", "index.html#autotoc_md91", null ]
      ] ]
    ] ],
    [ "Status Report", "index.html#autotoc_md93", null ],
    [ "Performance Notes", "index.html#autotoc_md95", null ],
    [ "Algorithm Theory", "index.html#autotoc_md97", null ],
    [ "2D Heat Equation", "page_heat_demo.html", [
      [ "Code", "page_heat_demo.html#autotoc_md50", null ],
      [ "Figure", "page_heat_demo.html#autotoc_md52", null ],
      [ "Library features used", "page_heat_demo.html#autotoc_md54", null ]
    ] ],
    [ "High-Performance Computing", "page_performance.html", [
      [ "Performance Limits", "page_performance.html#sec_roofline", [
        [ "Why Clock Frequency Plateaued", "page_performance.html#autotoc_md164", null ],
        [ "Amdahl's Law", "page_performance.html#autotoc_md165", null ],
        [ "Roofline Model", "page_performance.html#autotoc_md166", null ],
        [ "Arithmetic Intensity of Common Kernels", "page_performance.html#autotoc_md167", null ],
        [ "Memory Hierarchy Latencies", "page_performance.html#autotoc_md168", null ]
      ] ],
      [ "Cache-Blocked Matrix Multiplication", "page_performance.html#sec_cache_blocking", [
        [ "Why Naive \\f$ijk\\f$ Matmul is Slow", "page_performance.html#autotoc_md170", null ],
        [ "Worked Cache-Miss Count for N = 512", "page_performance.html#autotoc_md171", null ],
        [ "The Tiling Idea", "page_performance.html#autotoc_md172", null ],
        [ "Outer Loop Order: \\f$ii \\to jj \\to kk\\f$", "page_performance.html#autotoc_md173", null ],
        [ "Inner Loop Order: \\f$i \\to k \\to j\\f$", "page_performance.html#autotoc_md174", null ],
        [ "Measured Speedup", "page_performance.html#autotoc_md175", null ]
      ] ],
      [ "Register Blocking", "page_performance.html#sec_register_blocking", [
        [ "The Residual Problem After Cache Blocking", "page_performance.html#autotoc_md177", null ],
        [ "The Register Tile", "page_performance.html#autotoc_md178", null ],
        [ "Why Scalar Register Blocking Regresses", "page_performance.html#autotoc_md179", null ],
        [ "Register Blocking + SIMD as a Unit", "page_performance.html#autotoc_md180", null ]
      ] ],
      [ "SIMD Vectorization", "page_performance.html#sec_simd", [
        [ "AVX-256 and ARM NEON", "page_performance.html#autotoc_md182", null ],
        [ "The SIMD Micro-kernel", "page_performance.html#autotoc_md183", null ],
        [ "Compile-Time Architecture Dispatch", "page_performance.html#autotoc_md184", null ],
        [ "Jacobi SVD SIMD Fusion", "page_performance.html#autotoc_md185", null ],
        [ "Measured Progression", "page_performance.html#autotoc_md186", null ],
        [ "matvec_simd: Bandwidth Bound", "page_performance.html#autotoc_md187", null ]
      ] ]
    ] ],
    [ "Parallel and Distributed Computing", "page_parallel.html", [
      [ "MPI: Distributed Memory", "page_parallel.html#sec_mpi", [
        [ "The SPMD Model", "page_parallel.html#autotoc_md116", null ],
        [ "Point-to-Point Communication", "page_parallel.html#autotoc_md117", null ],
        [ "Collective Operations", "page_parallel.html#autotoc_md118", null ],
        [ "Distributed Dot Product", "page_parallel.html#autotoc_md119", null ],
        [ "Distributed Matrix-Vector Product", "page_parallel.html#autotoc_md120", null ],
        [ "Parallel Conjugate Gradient", "page_parallel.html#autotoc_md121", null ],
        [ "Hiding Latency with Non-Blocking Collectives", "page_parallel.html#autotoc_md122", null ]
      ] ],
      [ "CUDA: GPU Computing", "page_parallel.html#sec_cuda", [
        [ "GPU Hierarchy", "page_parallel.html#autotoc_md124", null ],
        [ "GPU Memory Hierarchy", "page_parallel.html#autotoc_md125", null ],
        [ "Coalesced Global Memory Access", "page_parallel.html#autotoc_md126", null ],
        [ "Tiled Matrix Multiplication with Shared Memory", "page_parallel.html#autotoc_md127", null ],
        [ "Kernel Launch Overhead and Crossover Points", "page_parallel.html#autotoc_md128", null ],
        [ "Batched Thomas Algorithm", "page_parallel.html#autotoc_md129", null ]
      ] ],
      [ "Banded Matrix Systems", "page_parallel.html#sec_banded", [
        [ "Band Structure", "page_parallel.html#autotoc_md131", null ],
        [ "LAPACK-Style Band Storage", "page_parallel.html#autotoc_md132", null ],
        [ "Banded LU with Partial Pivoting", "page_parallel.html#autotoc_md133", null ],
        [ "Numerical Stability: Growth Factor", "page_parallel.html#autotoc_md134", null ],
        [ "Solve Phase", "page_parallel.html#autotoc_md135", null ]
      ] ]
    ] ],
    [ "Fast Fourier Transform", "page_fft.html", [
      [ "The Discrete Fourier Transform", "page_fft.html#sec_dft", [
        [ "Interpretation of frequency bins", "page_fft.html#autotoc_md22", null ],
        [ "Parseval's theorem", "page_fft.html#autotoc_md23", null ]
      ] ],
      [ "Naive DFT: O(n^2) cost", "page_fft.html#sec_naive", null ],
      [ "Cooley-Tukey Radix-2 DIT", "page_fft.html#sec_cooley_tukey", [
        [ "Recurrence and complexity", "page_fft.html#autotoc_md26", null ],
        [ "Bit-reversal permutation", "page_fft.html#autotoc_md27", null ]
      ] ],
      [ "Algorithm (iterative DIT)", "page_fft.html#sec_algorithm", null ],
      [ "Real-to-Complex Transform (rfft)", "page_fft.html#sec_rfft", null ],
      [ "FFTPlan: amortizing planning cost", "page_fft.html#sec_fftplan", null ],
      [ "Backends", "page_fft.html#sec_backends", [
        [ "seq backend", "page_fft.html#autotoc_md32", null ],
        [ "fftw backend", "page_fft.html#autotoc_md33", null ]
      ] ],
      [ "API Reference", "page_fft.html#sec_api", null ],
      [ "Worked Example", "page_fft.html#sec_example", null ]
    ] ],
    [ "Week 6: Cache-Aware Computing -- Blocked Matrix Multiplication", "page_week6.html", [
      [ "Overview", "page_week6.html#autotoc_md231", null ],
      [ "1. The Memory Hierarchy", "page_week6.html#autotoc_md233", null ],
      [ "2. Arithmetic Intensity and the Roofline Model", "page_week6.html#autotoc_md235", null ],
      [ "3. Why Naive Matrix Multiply is Slow", "page_week6.html#autotoc_md237", [
        [ "Code (<tt>src/core/matrix.cpp:87-97</tt>)", "page_week6.html#autotoc_md238", null ]
      ] ],
      [ "4. The Blocking Idea", "page_week6.html#autotoc_md240", null ],
      [ "5. The Algorithm", "page_week6.html#autotoc_md242", [
        [ "Outer loop structure (<tt>src/core/matrix.cpp:153-161</tt>)", "page_week6.html#autotoc_md243", null ],
        [ "Micro-kernel (<tt>src/core/matrix.cpp:163-170</tt>)", "page_week6.html#autotoc_md244", null ]
      ] ],
      [ "6. Implementation Walkthrough", "page_week6.html#autotoc_md246", [
        [ "Declaration (<tt>include/core/matrix.hpp:48-64</tt>)", "page_week6.html#autotoc_md247", null ],
        [ "Full implementation (<tt>src/core/matrix.cpp:148-172</tt>)", "page_week6.html#autotoc_md248", null ]
      ] ],
      [ "7. Benchmark Results", "page_week6.html#autotoc_md250", null ],
      [ "8. Worked Cache-Miss Count", "page_week6.html#autotoc_md252", [
        [ "Naive loop – B accesses", "page_week6.html#autotoc_md253", null ],
        [ "Blocked loop – B accesses", "page_week6.html#autotoc_md254", null ]
      ] ],
      [ "9. Choosing the Block Size", "page_week6.html#autotoc_md256", null ],
      [ "10. What Comes Next", "page_week6.html#autotoc_md258", [
        [ "Step 1 – Register blocking", "page_week6.html#autotoc_md259", null ],
        [ "Step 2 – Explicit SIMD", "page_week6.html#autotoc_md260", null ]
      ] ],
      [ "11. Key Takeaways", "page_week6.html#autotoc_md262", null ],
      [ "Exercises", "page_week6.html#autotoc_md264", null ],
      [ "References", "page_week6.html#autotoc_md266", null ]
    ] ],
    [ "Week 7: Register Blocking -- What It Is and Why It Needs SIMD", "page_week7.html", [
      [ "Overview", "page_week7.html#autotoc_md267", null ],
      [ "1. The Residual Bottleneck After Cache Blocking", "page_week7.html#autotoc_md269", null ],
      [ "2. Register Blocking: The Idea", "page_week7.html#autotoc_md271", null ],
      [ "3. Implementation", "page_week7.html#autotoc_md273", [
        [ "Declaration (<tt>include/core/matrix.hpp:63-77</tt>)", "page_week7.html#autotoc_md274", null ],
        [ "Full implementation (<tt>src/core/matrix.cpp</tt>)", "page_week7.html#autotoc_md275", null ]
      ] ],
      [ "4. Benchmark Results – A Surprise", "page_week7.html#autotoc_md277", null ],
      [ "5. Why It Didn't Help: The Vectorisation Problem", "page_week7.html#autotoc_md279", [
        [ "What matmul_blocked does well", "page_week7.html#autotoc_md280", null ],
        [ "What register blocking does", "page_week7.html#autotoc_md281", null ],
        [ "The hardware angle", "page_week7.html#autotoc_md282", null ]
      ] ],
      [ "6. The Correct Mental Model: Register Blocking + SIMD Together", "page_week7.html#autotoc_md284", null ],
      [ "7. The BLAS Micro-kernel Design", "page_week7.html#autotoc_md286", null ],
      [ "8. What We Learned", "page_week7.html#autotoc_md288", null ],
      [ "9. Progression and Next Step", "page_week7.html#autotoc_md290", null ],
      [ "10. Key Takeaways", "page_week7.html#autotoc_md292", null ],
      [ "Exercises", "page_week7.html#autotoc_md294", null ],
      [ "References", "page_week7.html#autotoc_md296", null ]
    ] ],
    [ "Week 8: Explicit SIMD -- AVX-256 and ARM NEON", "page_week8.html", [
      [ "Overview", "page_week8.html#autotoc_md297", null ],
      [ "1. SIMD Fundamentals", "page_week8.html#autotoc_md299", null ],
      [ "2. The Micro-kernel", "page_week8.html#autotoc_md301", [
        [ "Why register blocking + SIMD work together", "page_week8.html#autotoc_md302", null ],
        [ "AVX-256 tile (<tt>src/core/matrix_simd.cpp</tt>)", "page_week8.html#autotoc_md303", null ],
        [ "ARM NEON tile (<tt>src/core/matrix_simd.cpp</tt>)", "page_week8.html#autotoc_md304", null ]
      ] ],
      [ "3. matvec_simd: Dot Product with Horizontal Reduction", "page_week8.html#autotoc_md306", [
        [ "AVX-256 version", "page_week8.html#autotoc_md307", null ],
        [ "NEON version", "page_week8.html#autotoc_md308", null ],
        [ "matvec benchmark results", "page_week8.html#autotoc_md309", null ]
      ] ],
      [ "4. Compile-Time Dispatch", "page_week8.html#autotoc_md311", [
        [ "Architecture detection (<tt>CMakeLists.txt</tt>)", "page_week8.html#autotoc_md312", null ],
        [ "Dispatch function (<tt>src/core/matrix_simd.cpp</tt>)", "page_week8.html#autotoc_md313", null ]
      ] ],
      [ "5. Boundary Handling", "page_week8.html#autotoc_md315", null ],
      [ "6. Complete Benchmark Results", "page_week8.html#autotoc_md317", [
        [ "Progression table", "page_week8.html#autotoc_md318", null ]
      ] ],
      [ "7. Key Takeaways", "page_week8.html#autotoc_md320", null ],
      [ "8. Exercises", "page_week8.html#autotoc_md322", null ],
      [ "References", "page_week8.html#autotoc_md324", null ]
    ] ],
    [ "NS Demo: From Slideshow to Real-Time", "page_ns_perf.html", [
      [ "The Solver in Brief", "page_ns_perf.html#sec_ns_overview", null ],
      [ "The Periodic Laplacian Stencil", "page_ns_perf.html#sec_stencil", null ],
      [ "Fix 1 – Boundary Peeling for NEON Auto-Vectorisation", "page_ns_perf.html#sec_fix_vectorise", [
        [ "The Problem", "page_ns_perf.html#autotoc_md101", null ],
        [ "The Fix", "page_ns_perf.html#autotoc_md102", null ],
        [ "Arithmetic Intensity", "page_ns_perf.html#autotoc_md103", null ]
      ] ],
      [ "Fix 2 – <tt>Backend::blas</tt> over <tt>Backend::omp</tt> for Cache-Resident Vectors", "page_ns_perf.html#sec_fix_policy", [
        [ "The Problem", "page_ns_perf.html#autotoc_md105", null ],
        [ "The Fix", "page_ns_perf.html#autotoc_md106", null ],
        [ "Backend Crossover Rule of Thumb", "page_ns_perf.html#autotoc_md107", null ]
      ] ],
      [ "Fix 3 – CG Tolerance and Warm-Starting", "page_ns_perf.html#sec_fix_cg", [
        [ "Spectral Condition Number of the Periodic Laplacian", "page_ns_perf.html#autotoc_md109", null ],
        [ "CG Convergence Rate", "page_ns_perf.html#autotoc_md110", null ],
        [ "Warm-Starting", "page_ns_perf.html#autotoc_md111", null ],
        [ "Periodic Poisson Singularity", "page_ns_perf.html#autotoc_md112", null ]
      ] ],
      [ "Combined Effect", "page_ns_perf.html#sec_ns_combined", null ],
      [ "Building and Running", "page_ns_perf.html#sec_ns_build", null ]
    ] ]
];