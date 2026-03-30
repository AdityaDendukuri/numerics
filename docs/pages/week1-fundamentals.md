# Week 1: Parallel Computing Fundamentals & Linear Algebra {#page_week1}

## 1. Why Parallel Computing?

Modern scientific computing demands computational power that single processors cannot deliver. Climate simulations, molecular dynamics, machine learning, and computational fluid dynamics all require billions of operations per second.

**The end of frequency scaling**: CPU clock speeds plateaued around 2005 at ~3-4 GHz due to power dissipation limits. The power consumed by a processor scales as:

\f[P \propto C V^2 f\f]

where \f$C\f$ is capacitance, \f$V\f$ is voltage, and \f$f\f$ is frequency. Higher frequencies require higher voltages, causing power to scale roughly as \f$f^3\f$.

The solution: **parallelism**. Instead of faster processors, we use *more* processors.

---

## 2. Taxonomy of Parallel Architectures

### Flynn's Taxonomy

| Type | Description | Example |
|------|-------------|---------|
| **SISD** | Single Instruction, Single Data | Classic uniprocessor |
| **SIMD** | Single Instruction, Multiple Data | GPU cores, vector units |
| **MIMD** | Multiple Instruction, Multiple Data | Multi-core CPUs, clusters |

### Memory Models

**Shared Memory**: All processors access a common address space.
```
+-----+ +-----+ +-----+ +-----+
| P0  | | P1  | | P2  | | P3  |
+--+--+ +--+--+ +--+--+ +--+--+
   |       |       |       |
   +-------+---+---+-------+
               |
        +------+------+
        |   Memory    |
        +-------------+
```

**Distributed Memory**: Each processor has private memory; data is exchanged via messages.
```
+---------+   +---------+   +---------+   +---------+
| P0 | M0 |   | P1 | M1 |   | P2 | M2 |   | P3 | M3 |
+----+----+   +----+----+   +----+----+   +----+----+
     |             |             |             |
     +-------------+------+------+-------------+
                          |
                    Network/Interconnect
```

---

## 3. Performance Limits

### Amdahl's Law

If a fraction \f$s\f$ of a program is inherently serial, the maximum speedup with \f$p\f$ processors is:

\f[S(p) = \frac{1}{s + \frac{1-s}{p}}\f]

As \f$p \to \infty\f$:

\f[S_{\max} = \frac{1}{s}\f]

**Example**: If 5% of your code is serial (\f$s = 0.05\f$), maximum speedup is \f$1/0.05 = 20\times\f$, regardless of processor count.

### Gustafson's Law

Amdahl assumes fixed problem size. In practice, we scale problem size with processor count. If we keep total runtime constant:

\f[S(p) = s + p(1-s) = p - s(p-1)\f]

This is more optimistic: speedup scales linearly with \f$p\f$ for fixed time.

### Roofline Model

Performance is bounded by either compute or memory bandwidth:

\f[\text{Attainable FLOP/s} = \min\left( \text{Peak FLOP/s},\ \text{Bandwidth} \times \text{Arithmetic Intensity} \right)\f]

**Arithmetic Intensity** \f$= \frac{\text{FLOPs}}{\text{Bytes transferred}}\f$

Operations with low arithmetic intensity are *memory-bound*; those with high intensity are *compute-bound*.

| Operation | Arithmetic Intensity | Bound |
|-----------|---------------------|-------|
| Vector addition | \f$\frac{1}{12}\f$ FLOP/byte | Memory |
| Dot product | \f$\frac{1}{8}\f$ FLOP/byte | Memory |
| Matrix-vector | \f$\frac{2n}{8(n+1)} \approx \frac{1}{4}\f$ | Memory |
| Matrix-matrix | \f$\frac{2n^3}{8 \cdot 3n^2} = \frac{n}{12}\f$ | Compute (for large \f$n\f$) |

---

## 4. Linear Algebra Fundamentals

### Vectors

A vector \f$\mathbf{x} \in \mathbb{R}^n\f$ is an ordered collection of \f$n\f$ real numbers:

\f[\mathbf{x} = \begin{pmatrix} x_1 \\ x_2 \\ \vdots \\ x_n \end{pmatrix}\f]

**Vector Operations**:

| Operation | Definition | FLOPs | Memory |
|-----------|------------|-------|--------|
| Scale | \f$\mathbf{y} = \alpha \mathbf{x}\f$ | \f$n\f$ | \f$2n\f$ |
| Addition | \f$\mathbf{z} = \mathbf{x} + \mathbf{y}\f$ | \f$n\f$ | \f$3n\f$ |
| AXPY | \f$\mathbf{y} = \alpha\mathbf{x} + \mathbf{y}\f$ | \f$2n\f$ | \f$3n\f$ |
| Dot product | \f$\alpha = \mathbf{x}^T \mathbf{y} = \sum_{i=1}^n x_i y_i\f$ | \f$2n\f$ | \f$2n\f$ |
| Norm | \f$\|\mathbf{x}\|_2 = \sqrt{\mathbf{x}^T \mathbf{x}}\f$ | \f$2n\f$ | \f$n\f$ |

### Matrices

A matrix \f$A \in \mathbb{R}^{m \times n}\f$ is a 2D array with \f$m\f$ rows and \f$n\f$ columns:

\f[A = \begin{pmatrix} a_{11} & a_{12} & \cdots & a_{1n} \\ a_{21} & a_{22} & \cdots & a_{2n} \\ \vdots & \vdots & \ddots & \vdots \\ a_{m1} & a_{m2} & \cdots & a_{mn} \end{pmatrix}\f]

**Storage Layouts**:

*Row-major* (C/C++): Elements in each row are contiguous.
```
A[i][j] -> memory[i * n + j]
```

*Column-major* (Fortran): Elements in each column are contiguous.
```
A[i][j] -> memory[j * m + i]
```

### Matrix-Vector Multiplication

Given \f$A \in \mathbb{R}^{m \times n}\f$ and \f$\mathbf{x} \in \mathbb{R}^n\f$, compute \f$\mathbf{y} = A\mathbf{x}\f$ where \f$\mathbf{y} \in \mathbb{R}^m\f$:

\f[y_i = \sum_{j=1}^{n} a_{ij} x_j \quad \text{for } i = 1, \ldots, m\f]

**Complexity**: \f$2mn\f$ FLOPs, \f$O(mn)\f$ memory access.

```cpp
for (int i = 0; i < m; ++i) {
    y[i] = 0;
    for (int j = 0; j < n; ++j) {
        y[i] += A[i * n + j] * x[j];
    }
}
```

### Matrix-Matrix Multiplication

Given \f$A \in \mathbb{R}^{m \times k}\f$ and \f$B \in \mathbb{R}^{k \times n}\f$, compute \f$C = AB\f$ where \f$C \in \mathbb{R}^{m \times n}\f$:

\f[c_{ij} = \sum_{\ell=1}^{k} a_{i\ell} b_{\ell j}\f]

**Complexity**: \f$2mnk\f$ FLOPs.

For square matrices (\f$m = k = n\f$), this is \f$O(n^3)\f$ FLOPs with \f$O(n^2)\f$ data--hence *compute-bound* for large \f$n\f$.

```cpp
for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
        C[i * n + j] = 0;
        for (int l = 0; l < k; ++l) {
            C[i * n + j] += A[i * k + l] * B[l * n + j];
        }
    }
}
```

---

## 5. Parallelizing Linear Algebra

### Data Decomposition

**1D Block Distribution**: Partition vector/matrix rows among processors.

For vector \f$\mathbf{x}\f$ of length \f$n\f$ across \f$p\f$ processors:
- Processor \f$i\f$ owns elements \f$[i \cdot n/p,\ (i+1) \cdot n/p)\f$

**2D Block Distribution**: Partition matrix into rectangular tiles.

For matrix \f$A\f$ on a \f$p_r \times p_c\f$ processor grid:
- Processor \f$(i, j)\f$ owns a block of size \f$(m/p_r) \times (n/p_c)\f$

### Parallel Dot Product

Each processor computes a local partial sum, then a global reduction combines results:

\f[\mathbf{x}^T \mathbf{y} = \sum_{i=0}^{p-1} \left( \sum_{j \in \text{local}_i} x_j y_j \right)\f]

```
Processor 0:  x[0:249]*y[0:249]   = 125.0  -+
Processor 1:  x[250:499]*y[250:499] = 130.0  +--> Reduce(+) -> 510.0
Processor 2:  x[500:749]*y[500:749] = 128.0  |
Processor 3:  x[750:999]*y[750:999] = 127.0  -+
```

### Parallel Matrix-Vector Product

For \f$\mathbf{y} = A\mathbf{x}\f$ with row-distributed \f$A\f$:

1. Each processor needs the *entire* vector \f$\mathbf{x}\f$
2. Broadcast or all-gather \f$\mathbf{x}\f$
3. Each processor computes its local rows of \f$\mathbf{y}\f$

**Communication cost**: \f$O(n)\f$ per processor for the gather.

### Parallel Matrix-Matrix Product

For \f$C = AB\f$ on a 2D processor grid, use algorithms like:
- **Cannon's Algorithm**: Shift-based, memory-efficient
- **SUMMA**: Broadcast-based, simpler to implement
- **2.5D Algorithms**: Trade memory for reduced communication

---

## 6. Key Takeaways

1. **Parallelism is essential**: Clock speeds won't save us; we must use multiple processors.

2. **Know your bounds**: Use Amdahl/Gustafson to estimate potential speedup; use Roofline to identify bottlenecks.

3. **Arithmetic intensity matters**: Matrix-matrix multiplication parallelizes well; dot products are memory-bound.

4. **Communication is expensive**: Minimizing data movement is often more important than minimizing computation.

5. **Data layout affects performance**: Row-major vs. column-major, cache utilization, and memory access patterns are critical.

---

## Exercises

1. A program spends 10% of its time in serial code. What is the maximum speedup with (a) 10 processors? (b) 100 processors? (c) infinite processors?

2. Compute the arithmetic intensity of the AXPY operation \f$\mathbf{y} = \alpha \mathbf{x} + \mathbf{y}\f$.

3. For a 1000x1000 matrix-matrix multiplication, how many FLOPs are performed? If your machine achieves 100 GFLOP/s, what is the minimum execution time?

4. You have a vector of length \f$10^6\f$ distributed across 4 processors. Describe the communication pattern needed to compute the Euclidean norm.

---

## References

- Hennessy & Patterson, *Computer Architecture: A Quantitative Approach*
- Golub & Van Loan, *Matrix Computations*
- Williams, Waterman & Patterson, "Roofline: An Insightful Visual Performance Model"
