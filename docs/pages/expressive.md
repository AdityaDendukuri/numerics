# Value-Returning Expression Interface {#page_expressive}

For rapid mathematical prototyping, concise unit testing, and formula readability, Numerics provides an opt-in convenience expression tier.

---

## 1. Overview

While the core `numerics` architecture is strictly designed around zero-allocation, out-parameter kernels (such as `matvec(A, x, y)` and `axpy(a, x, y)`), mathematical prototyping and test assertions often benefit from natural value-returning infix expressions.

The convenience layer provides:
1. **Convenience mat & vec Constructors:** `zeros`, `ones`, `eye`, `linspace`, `accu`.
2. **Infix Operator Overloads:** `+`, `-`, `*`, `/` over matrices, vectors, and scalars via `using namespace num::ops;`.
3. **Value-Returning Arithmetic:** Functions that allocate and return results directly.

---

## 2. Constructors & Utility Functions

```cpp
#include <numerics.hpp>

// mat constructors
num::mat Z = num::zeros(4, 4);      // 4x4 matrix initialized to 0.0
num::mat O = num::ones(3, 2);       // 3x2 matrix initialized to 1.0
num::mat I = num::eye(4);           // 4x4 identity matrix
num::mat E = num::eye(3, 5);        // 3x5 rectangular identity-like matrix

// vec constructors
num::vec vz = num::zeros(5);        // Length-5 zero vector
num::vec vo = num::ones(3);         // Length-3 ones vector
num::vec v  = num::linspace(0, 1, 5);// [0.0, 0.25, 0.5, 0.75, 1.0]

// Reductions
num::real s1 = num::accu(Z);           // Sum of all elements in matrix
num::real s2 = num::accu(v);           // Sum of all elements in vector
```

---

## 3. Infix Operator Overloads (num::ops)

To avoid polluting the global or `num` namespace with unintended operator overloads, algebraic operators are isolated within the `num::ops` namespace:

```cpp
#include <numerics.hpp>

using namespace num::ops;

num::mat A = num::ones(3, 3);
num::mat B = num::eye(3);
num::vec x = num::linspace(1.0, 3.0, 3);

// mat-matrix multiplication and addition
num::mat C = A * B + 2.0 * B;

// mat-vector multiplication and scaling
num::vec y = A * x - x / 2.0;

// vec arithmetic
num::vec z = 2.0 * x + y;
```

---

## 4. Why Value-Returning Expressions Are Not Preferable in Production Code

While value-returning infix syntax is intuitive for small scripts and unit tests, it is **strongly discouraged in performance-critical numerical simulation loops, ODE time-steppers, and iterative solver kernels** for the following reasons:

### 1. Hidden Dynamic Memory Allocations
Every binary operator creates and returns a newly heap-allocated object (`new double[n]`):
```cpp
// Evaluating this expression:
num::vec r = b - A * x - 0.5 * C * x;

// Results in:
// 1. Heap allocation for (A * x)
// 2. Heap allocation for (C * x)
// 3. Heap allocation for (0.5 * (C * x))
// 4. Heap allocation for (b - (A * x))
// 5. Heap allocation for final subtraction
// Followed by 5 separate heap deallocations!
```
In a numerical time-stepper running for \f$10^6\f$ iterations, this generates millions of dynamic allocations, resulting in:
* Severe memory allocator lock contention.
* Cache line invalidation and memory fragmentation.
* High latency overhead compared to register-tiled in-place execution.

### 2. Lack of Loop Fusion
Without complex lazy expression template trees (which drastically increase compilation times and can hinder compiler vectorization), each sub-expression must be evaluated into a temporary memory buffer before the next operator can execute. This produces multiple memory read/write sweeps across RAM instead of a single fused vectorized pass.

### 3. Bypassing Mathematical Invariant Verification
Direct uncertified convenience routines (such as unconstrained `solve(A, b)` or `inv(A)`) treat matrices as general dense arrays. They cannot exploit structural guarantees—such as positive-definiteness, bandedness, or symmetry—that allow `numerics` to dispatch specialized \f$\mathcal{O}(n)\f$ or \f$\mathcal{O}(n^2)\f$ solvers.

---

## 5. The Recommended Idiom: Pre-Allocated Out-Parameter Kernels

In production code and simulation loops, allocate destination buffers once outside the loop and reuse them with mutating kernels:

```cpp
// Pre-allocate once outside the hot loop
num::vec y(n, 0.0);
num::vec tmp(n, 0.0);

for (num::idx step = 0; step < total_steps; ++step) {
    // Zero dynamic allocations inside the loop
    num::matvec(A, x, y);           // y = A * x
    num::axpy(2.0, z, y);           // y = y + 2.0 * z
    num::cholesky_solve(L, b, x);   // In-place solver with verified SPD evidence
}
```

---

## Summary Comparison

| Criterion | Value-Returning Expressions (`num::ops`) | Preferred Idiomatic (`num::matvec`, `num::axpy`) |
| :--- | :--- | :--- |
| **Readability** | High (natural mathematical formulas) | Explicit (out-parameter buffers) |
| **Heap Allocations** | Allocates on every operator call | **Zero** heap allocations in loops |
| **Performance** | Memory-bandwidth and allocator bound | Peak hardware FLOPS & SIMD throughput |
| **Best Used For** | Unit tests, quick scripts, textbook verification | Production simulations, ODE steppers, solvers |
