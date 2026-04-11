# NS Demo: From Slideshow to Real-Time {#page_ns_perf}

The 2D Navier-Stokes stress test (`ns_demo`) solves the incompressible
Navier-Stokes equations on a periodic MAC grid using Chorin's projection method,
visualised in real-time with raylib.  The initial implementation ran too slowly to
be interactive on a 256x256 grid.  Three focused fixes -- each traceable to a
single hardware-software mismatch -- recovered real-time performance without
changing the algorithm.

---

## The Solver in Brief {#sec_ns_overview}

Each timestep executes three phases:

1. **Semi-Lagrangian advection** -- backtrace each MAC face along the velocity
   field, bilinearly interpolate the transported value.
2. **Build right-hand side** -- compute the discrete divergence of the intermediate
   velocity field \f$\mathbf{u}^*\f$.
3. **Pressure projection** -- solve the periodic Poisson equation, then subtract
   the pressure gradient to enforce \f$\nabla \cdot \mathbf{u} = 0\f$.

The Poisson solve is the inner loop: it calls @ref num::cg_matfree with a
matrix-free stencil operator as the matvec, issuing hundreds of dot products and
AXPY operations per frame.  Everything else is \f$O(N^2)\f$ per timestep; the
pressure solve is where time is spent.

---

## The Periodic Laplacian Stencil {#sec_stencil}

The matvec for the matrix-free CG applies the negative discrete Laplacian on a
periodic \f$N \times N\f$ grid:

\f[
(-\Delta p)_{i,j} = \frac{1}{h^2}\bigl(
  4\,p_{i,j}
  - p_{i+1,j} - p_{i-1,j}
  - p_{i,j+1} - p_{i,j-1}
\bigr)
\f]

All indices are taken modulo \f$N\f$ (periodic boundary conditions).  This
operator is positive semi-definite on the zero-mean subspace, which is exactly
what CG requires.

---

## Fix 1 -- Boundary Peeling for NEON Auto-Vectorisation {#sec_fix_vectorise}

### The Problem

The naive inner loop inside the stencil operator was:

```cpp
for (idx j = 0; j < N; ++j)
    d[j] = inv_h2 * (4*row[j] - row_p[j] - row_m[j]
                   - row[wp1(j)] - row[wm1(j)]);
```

`wp1(j)` and `wm1(j)` call `(j+1) % N` and `(j+N-1) % N` respectively.  Even
though the modulo only matters at `j = 0` and `j = N-1`, the compiler cannot
prove that at the vectorisation stage.  On Apple Silicon the inner loop must
lower to ARM NEON `vfmaq_f64` instructions that operate on two `double`s
simultaneously -- but that requires the gather indices `wp1(j)` and `wm1(j)` to
be predictable, unit-stride offsets.  Modulo calls break that guarantee.

**Result**: the compiler falls back to scalar code.  The stencil runs at 1/2 the
arithmetic throughput available to the hardware.

### The Fix

Peel the two boundary iterations out of the inner loop, leaving `j = 1..N-2`
with no modulo calls:

```cpp
// j = 0: wrap left  -> j-1 = N-1
d[0] = inv_h2 * (4*row[0] - row_p[0] - row_m[0]
               - row[1] - row[N-1]);

// j = 1..N-2: unit-stride access -- compiler vectorises with NEON
for (idx j = 1; j < N - 1; ++j)
    d[j] = inv_h2 * (4*row[j] - row_p[j] - row_m[j]
                   - row[j+1] - row[j-1]);

// j = N-1: wrap right -> j+1 = 0
d[N-1] = inv_h2 * (4*row[N-1] - row_p[N-1] - row_m[N-1]
                 - row[0]   - row[N-2]);
```

The inner loop now accesses five contiguous arrays with fixed offsets \f$-1, 0,
+1\f$ -- a classic 1-D stencil pattern.  Compiled with `-O3 -march=native`, clang
emits `ld1` loads and `fmla` (fused multiply-accumulate) instructions over
pairs of doubles.

### Arithmetic Intensity

The five-point stencil reads five doubles per output element and writes one:

\f[
I = \frac{6 \text{ FLOPs}}{6 \times 8 \text{ bytes}} = \frac{1}{8} \text{ FLOP/byte}
\f]

This is solidly **memory-bandwidth bound** -- the same category as dot product.
Vectorisation does not change the FLOP count; it cuts the **instruction count**
in half (two doubles per NEON lane), which reduces loop overhead and frees
execution units.  The practical gain is approximately 1.5-2x on Apple M-series
because the M-core has a generous load bandwidth that keeps NEON fed.

The outer `i` loop is parallelised with `#pragma omp parallel for schedule(static)`,
so all \f$N\f$ rows run concurrently across performance cores.

---

## Fix 2 -- `Backend::blas` over `Backend::omp` for Cache-Resident Vectors {#sec_fix_policy}

### The Problem

The CG inner loop consists of six vector operations per iteration -- two dot
products, three AXPY, one norm -- all over vectors of length \f$N^2\f$.  The
original code used `num::omp` (`Backend::omp`):

```cpp
auto alpha = num::dot(r, r, num::omp) / num::dot(Ap, p, num::omp);
num::axpy(alpha,  p,  u_star, num::omp);   // u <- u + alpha p
num::axpy(-alpha, Ap, r,      num::omp);   // r <- r - alpha Ap
```

For \f$N = 256\f$ the vectors are \f$256^2 = 65\,536\f$ doubles, occupying
**512 KB**.  The M-series L2 cache is 12 MB per cluster; all three active CG
vectors fit comfortably.

The problem is **thread fork/join overhead**.  `num::omp` spawns a thread team
(or wakes a parked team) for each operation.  OpenMP barrier synchronisation on
macOS/Apple Silicon costs roughly 5-15 mus per fork -- the same order of magnitude
as the vector operation itself at this size.

### The Fix

Switch the CG kernel to `num::blas` (`Backend::blas`, Accelerate framework):

```cpp
auto alpha = num::dot(r, r, num::blas) / num::dot(Ap, p, num::blas);
num::axpy(alpha,  p,  u_star, num::blas);
num::axpy(-alpha, Ap, r,      num::blas);
```

Apple's Accelerate `cblas_ddot` and `cblas_daxpy` are single-call,
internally-threaded routines that avoid user-space fork overhead.  For
cache-resident vectors Accelerate typically uses NEON with software pipelining
optimised for the specific Apple Silicon memory subsystem.

### Backend Crossover Rule of Thumb

| Vector size | Working set | Recommended backend |
|-------------|------------|-------------------|
| \f$n \leq 2^{16}\f$ | <= 512 KB -- fits in L2 | `Backend::blas` (Accelerate/MKL) |
| \f$2^{16} < n \leq 2^{20}\f$ | 512 KB - 8 MB | either; benchmark on target |
| \f$n > 2^{20}\f$ | > 8 MB -- spills to L3/DRAM | `Backend::omp` amortises fork |

The crossover is hardware-dependent.  On Linux with glibc pthreads and a small
thread pool, `num::omp` fork overhead is typically lower and the crossover shifts
toward smaller vectors.

---

## Fix 3 -- CG Tolerance and Warm-Starting {#sec_fix_cg}

### Spectral Condition Number of the Periodic Laplacian

The discrete negative Laplacian on an \f$N \times N\f$ periodic grid has
eigenvalues

\f[
\lambda_{k,l} = \frac{4}{h^2}\Bigl(
  \sin^2\!\frac{\pi k}{N} + \sin^2\!\frac{\pi l}{N}
\Bigr), \quad k,l = 0,\ldots,N-1
\f]

restricted to the zero-mean subspace (the \f$k = l = 0\f$ DC mode is removed).
The minimum non-zero eigenvalue is

\f[
\lambda_{\min} = \frac{4}{h^2}\sin^2\!\frac{\pi}{N}
\approx \frac{4\pi^2}{N^2 h^2} = 4\pi^2
\quad (h = 1/N)
\f]

and the maximum is \f$\lambda_{\max} = 8/h^2 = 8N^2\f$, giving spectral
condition number

\f[
\kappa = \frac{\lambda_{\max}}{\lambda_{\min}}
       \approx \frac{8N^2}{4\pi^2}
       = \frac{2N^2}{\pi^2}
\f]

For \f$N = 256\f$: \f$\kappa \approx 13\,300\f$.

### CG Convergence Rate

For an SPD system with condition number \f$\kappa\f$, CG satisfies

\f[
\frac{\|\mathbf{e}_k\|_A}{\|\mathbf{e}_0\|_A}
\leq 2\left(\frac{\sqrt{\kappa}-1}{\sqrt{\kappa}+1}\right)^k
\f]

To reduce the error by a factor of \f$10^{-4}\f$ from the initial residual,
the number of iterations required is approximately

\f[
k \geq \frac{\ln(2 \times 10^4)}{\ln\!\left(\dfrac{\sqrt{\kappa}+1}{\sqrt{\kappa}-1}\right)}
\approx \frac{\sqrt{\kappa}}{2} \ln(2 \times 10^4)
\approx \frac{115}{2} \times 9.9 \approx 570
\f]

With tolerance `1e-3` the target drops to \f$10^{-3}\f$, requiring roughly
\f$\sqrt{\kappa}/2 \times \ln(2000) \approx 430\f$ iterations -- still large.

### Warm-Starting

The pressure field changes very little between
consecutive timesteps.  Initialising each CG solve with \f$p^{n-1}\f$ rather
than zero places the initial residual already close to the solution.  The
practical initial relative residual is typically below \f$10^{-2}\f$ after the
first few timesteps, cutting the number of iterations needed to below
\f$\sim 100\f$.

The solver caps iterations at `max_iter = 100`.  For a well-started warm solve at
\f$N = 256\f$, convergence to `tol = 1e-3` typically requires 60-90 iterations.

### Periodic Poisson Singularity

The constant function is in the null space of the Laplacian, so the right-hand
side must be zero-mean for a solution to exist.  Before the solve, the mean of
the RHS is removed:

\f[
b \leftarrow b - \bar{b}, \quad \bar{b} = \frac{1}{N^2}\sum_{i,j} b_{i,j}
\f]

After convergence the mean of the solution is similarly removed.  This keeps the
pressure in the correct (zero-mean) invariant subspace and prevents CG from
chasing the null-space component, which would stall convergence.

---

## Combined Effect {#sec_ns_combined}

The three fixes act on independent bottlenecks:

| Fix | Root cause | Mechanism | Observed gain |
|-----|-----------|-----------|--------------|
| Boundary peeling | Modulo in inner loop blocks NEON | Enables auto-vectorisation | ~1.8x stencil throughput |
| `Backend::blas` | OMP fork overhead >> vector op time | Eliminates thread sync | ~2x per CG vector op |
| Warm-start + loose tolerance | CG initialised far from solution | Reduces iteration count | ~3-5x fewer CG iters/frame |

No single fix dominates; the product of all three converts a slideshow into a
simulation that runs comfortably above 30 FPS at \f$N = 256\f$ on Apple Silicon,
with the pressure solve consuming 5-15 ms per frame instead of 150+ ms.

The lesson is a recurring one in high-performance computing: before adding
algorithmic complexity, confirm that the baseline is not simply leaving hardware
throughput on the floor through avoidable penalties -- modulo calls in hot loops,
mismatched parallelism granularity, and unnecessary solver restarts are all
examples of exactly that.

---

## Building and Running {#sec_ns_build}

```bash
cmake --preset app-ns
cmake --build --preset app-ns
./build/apps/ns_demo/ns_demo        # default: N=256
./build/apps/ns_demo/ns_demo 512    # larger grid
```

**Controls** -- `SPACE` pause/resume, `R` reset to shear layer IC, `+`/`-`
adjust substeps per frame, `[`/`]` scale vorticity colormap, `ESC` quit.

The HUD reports CG iteration count, final residual, and per-phase wall time each
frame so the solver behaviour can be observed in real time.
