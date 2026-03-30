# Week 2: MPI -- Distributed Memory Programming {#page_week2}

## 1. The Message Passing Model

MPI (Message Passing Interface) is the standard for distributed-memory parallel programming. Each process has its own private memory; data is exchanged explicitly via messages.

```
+--------------+    message     +--------------+
|   Process 0  | ------------->  |   Process 1  |
|  +--------+  |                |  +--------+  |
|  | Memory |  |                |  | Memory |  |
|  +--------+  |  <-------------  |  +--------+  |
+--------------+    message     +--------------+
```

**Key characteristics**:
- Processes are independent programs (SPMD: Single Program, Multiple Data)
- No shared state--all communication is explicit
- Scales to thousands of nodes across networks

---

## 2. MPI Basics

### Initialization and Finalization

Every MPI program begins with `MPI_Init` and ends with `MPI_Finalize`:

```cpp
#include <mpi.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);  // My process ID
    MPI_Comm_size(MPI_COMM_WORLD, &size);  // Total processes

    // ... parallel code ...

    MPI_Finalize();
    return 0;
}
```

**Compile and run**:
```bash
mpicxx -o program program.cpp
mpirun -np 4 ./program
```

### Communicators

A **communicator** defines a group of processes that can communicate. `MPI_COMM_WORLD` includes all processes. You can create subgroups for hierarchical algorithms.

| Function | Description |
|----------|-------------|
| `MPI_Comm_rank(comm, &rank)` | Get process ID within communicator |
| `MPI_Comm_size(comm, &size)` | Get number of processes |
| `MPI_Comm_split(...)` | Create sub-communicators |

---

## 3. Point-to-Point Communication

### Blocking Send/Receive

The most basic operations: `MPI_Send` and `MPI_Recv`.

```cpp
// Process 0 sends to Process 1
if (rank == 0) {
    double data = 3.14;
    MPI_Send(&data, 1, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD);
    //       buffer, count, type, dest, tag, comm
}
else if (rank == 1) {
    double data;
    MPI_Recv(&data, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    //       buffer, count, type, source, tag, comm, status
}
```

**Parameters**:
- `count`: Number of elements (not bytes)
- `tag`: Message identifier (use 0 if you don't need it)
- `source`/`dest`: Rank of partner process

### Common MPI Datatypes

| MPI Type | C Type |
|----------|--------|
| `MPI_CHAR` | `char` |
| `MPI_INT` | `int` |
| `MPI_LONG` | `long` |
| `MPI_FLOAT` | `float` |
| `MPI_DOUBLE` | `double` |

### Deadlock

**Deadlock** occurs when processes wait for each other indefinitely:

```cpp
// DEADLOCK: Both processes wait to receive before sending
if (rank == 0) {
    MPI_Recv(..., 1, ...);  // Waits for rank 1
    MPI_Send(..., 1, ...);
} else {
    MPI_Recv(..., 0, ...);  // Waits for rank 0
    MPI_Send(..., 0, ...);
}
```

**Solution**: Use `MPI_Sendrecv` or non-blocking operations:

```cpp
// Safe exchange using Sendrecv
MPI_Sendrecv(&send_data, 1, MPI_DOUBLE, partner, 0,
             &recv_data, 1, MPI_DOUBLE, partner, 0,
             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
```

### Non-Blocking Communication

Non-blocking calls return immediately; use `MPI_Wait` to complete:

```cpp
MPI_Request request;
MPI_Isend(&data, n, MPI_DOUBLE, dest, tag, comm, &request);

// ... do other work while message is in flight ...

MPI_Wait(&request, MPI_STATUS_IGNORE);  // Block until complete
```

This enables **overlapping computation and communication**.

---

## 4. Collective Operations

Collectives involve all processes in a communicator. They are optimized and easier to use than manual point-to-point patterns.

### Broadcast

One process sends data to all others:

```
Before:  [A B C D]  [ - - - - ]  [ - - - - ]  [ - - - - ]
            down           down           down           down
After:   [A B C D]  [A B C D]   [A B C D]   [A B C D]
```

```cpp
double data[4];
if (rank == 0) {
    // Initialize data on root
}
MPI_Bcast(data, 4, MPI_DOUBLE, 0, MPI_COMM_WORLD);
//        buffer, count, type, root, comm
```

### Reduce

Combine data from all processes using an operation:

```
Process 0: [3]  -+
Process 1: [1]  -+--> MPI_SUM -> [10] on root
Process 2: [4]  -+
Process 3: [2]  -+
```

```cpp
double local_sum = compute_partial();
double global_sum;
MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
//         send_buf, recv_buf, count, type, op, root, comm
```

**Reduction operations**: `MPI_SUM`, `MPI_PROD`, `MPI_MAX`, `MPI_MIN`, `MPI_LAND`, `MPI_LOR`

### Allreduce

Like `Reduce`, but result goes to all processes:

```cpp
MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
// Now every process has the same global value
```

### Gather and Scatter

**Gather**: Collect data from all processes to root:
```
Before:  [A]        [B]        [C]        [D]
           ->         down         ->
After:          [A B C D] on root
```

**Scatter**: Distribute data from root to all:
```
Before:      [A B C D] on root
               -> down ->
After:   [A]        [B]        [C]        [D]
```

```cpp
// Gather
double local_val = rank * 1.0;
double* gathered = (rank == 0) ? new double[size] : nullptr;
MPI_Gather(&local_val, 1, MPI_DOUBLE, gathered, 1, MPI_DOUBLE, 0, comm);

// Scatter
double recv_val;
MPI_Scatter(gathered, 1, MPI_DOUBLE, &recv_val, 1, MPI_DOUBLE, 0, comm);
```

### Allgather

Every process gets the gathered result:

```cpp
double local = rank;
double all_data[size];
MPI_Allgather(&local, 1, MPI_DOUBLE, all_data, 1, MPI_DOUBLE, comm);
// Now every process has [0, 1, 2, 3, ...]
```

### Summary Table

| Collective | Description | Complexity |
|------------|-------------|------------|
| `MPI_Bcast` | One-to-all | \f$O(\log p)\f$ |
| `MPI_Reduce` | All-to-one with operation | \f$O(\log p)\f$ |
| `MPI_Allreduce` | Reduce + Broadcast | \f$O(\log p)\f$ |
| `MPI_Gather` | Collect to root | \f$O(p)\f$ |
| `MPI_Scatter` | Distribute from root | \f$O(p)\f$ |
| `MPI_Allgather` | Gather + Broadcast | \f$O(p)\f$ |
| `MPI_Alltoall` | Complete exchange | \f$O(p)\f$ |

---

## 5. Distributed Linear Algebra with MPI

### Distributed Vector

Partition a vector of length \f$n\f$ across \f$p\f$ processes:

```cpp
int local_n = n / size;  // Assume n divisible by size
std::vector<double> local_x(local_n);

// Each process owns x[rank*local_n : (rank+1)*local_n]
```

### Distributed Dot Product

\f[\mathbf{x}^T \mathbf{y} = \sum_{i=0}^{p-1} \underbrace{\sum_{j \in \text{local}_i} x_j y_j}_{\text{local dot}}\f]

```cpp
double local_dot = 0.0;
for (int i = 0; i < local_n; ++i) {
    local_dot += local_x[i] * local_y[i];
}

double global_dot;
MPI_Allreduce(&local_dot, &global_dot, 1, MPI_DOUBLE, MPI_SUM, comm);
```

**Communication**: One `Allreduce` = \f$O(\log p)\f$ messages.

### Distributed Norm

\f[\|\mathbf{x}\|_2 = \sqrt{\sum_i x_i^2} = \sqrt{\text{Allreduce}\left(\sum_{j \in \text{local}} x_j^2\right)}\f]

```cpp
double local_sq = 0.0;
for (int i = 0; i < local_n; ++i) {
    local_sq += local_x[i] * local_x[i];
}

double global_sq;
MPI_Allreduce(&local_sq, &global_sq, 1, MPI_DOUBLE, MPI_SUM, comm);
double norm = std::sqrt(global_sq);
```

### Distributed Matrix-Vector Product

For \f$\mathbf{y} = A\mathbf{x}\f$ with row-distributed \f$A\f$:

```
Process 0: A[0:m/p, :]     needs all of x    -> computes y[0:m/p]
Process 1: A[m/p:2m/p, :]  needs all of x    -> computes y[m/p:2m/p]
...
```

**Algorithm**:
1. `Allgather` the distributed vector \f$\mathbf{x}\f$
2. Each process computes its local rows

```cpp
// x is distributed, we need full x for matvec
std::vector<double> full_x(n);
MPI_Allgather(local_x.data(), local_n, MPI_DOUBLE,
              full_x.data(), local_n, MPI_DOUBLE, comm);

// Compute local portion of y = A * x
for (int i = 0; i < local_m; ++i) {
    local_y[i] = 0.0;
    for (int j = 0; j < n; ++j) {
        local_y[i] += local_A[i * n + j] * full_x[j];
    }
}
```

**Communication cost**: \f$O(n)\f$ for the `Allgather`.

---

## 6. Performance Considerations

### Communication vs. Computation

Let \f$t_s\f$ = latency (startup time), \f$t_w\f$ = time per word.

**Point-to-point**: \f$T = t_s + n \cdot t_w\f$

**Broadcast** (tree-based): \f$T = t_s \log p + n \cdot t_w \log p\f$

**Allreduce**: \f$T = 2 t_s \log p + 2n \cdot t_w\f$

### Hiding Latency

Use non-blocking collectives (MPI-3):

```cpp
MPI_Request req;
MPI_Iallreduce(&local, &global, 1, MPI_DOUBLE, MPI_SUM, comm, &req);

// Do independent computation while reduction proceeds
do_local_work();

MPI_Wait(&req, MPI_STATUS_IGNORE);
```

### Load Balancing

Uneven work distribution kills parallel efficiency. For \f$n\f$ not divisible by \f$p\f$:

```cpp
int local_n = n / size;
int remainder = n % size;
if (rank < remainder) local_n++;  // First 'remainder' ranks get one extra
```

---

## 7. Example: Parallel Conjugate Gradient

The CG algorithm for solving \f$A\mathbf{x} = \mathbf{b}\f$ uses only:
- Matrix-vector products
- Dot products
- Vector updates (AXPY)

```cpp
// Pseudocode for parallel CG
r = b - A*x;           // matvec + local ops
p = r;
rsold = dot(r, r);     // Allreduce

for (int i = 0; i < max_iter; ++i) {
    Ap = A * p;            // Allgather + local matvec
    alpha = rsold / dot(p, Ap);  // Allreduce

    x = x + alpha * p;     // Local AXPY
    r = r - alpha * Ap;    // Local AXPY

    rsnew = dot(r, r);     // Allreduce

    if (sqrt(rsnew) < tol) break;

    p = r + (rsnew/rsold) * p;  // Local ops
    rsold = rsnew;
}
```

**Communication per iteration**: 2 Allreduce + 1 Allgather.

---

## 8. Our Library's MPI Interface

The `num::mpi` namespace provides:

```cpp
namespace num::mpi {
    void init(int* argc, char*** argv);
    void finalize();

    int rank(MPI_Comm comm = MPI_COMM_WORLD);
    int size(MPI_Comm comm = MPI_COMM_WORLD);

    // Distributed dot product (each rank holds partial vector)
    real dot(const Vector& x, const Vector& y, MPI_Comm comm = MPI_COMM_WORLD);

    // Distributed norm
    real norm(const Vector& x, MPI_Comm comm = MPI_COMM_WORLD);

    // Collective wrappers
    void allreduce_sum(real* data, idx n, MPI_Comm comm = MPI_COMM_WORLD);
    void broadcast(real* data, idx n, int root = 0, MPI_Comm comm = MPI_COMM_WORLD);
}
```

**Usage**:
```cpp
#include "numerics/numerics.hpp"

int main(int argc, char** argv) {
    num::mpi::init(&argc, &argv);

    int local_n = 1000;
    num::Vector x(local_n, 1.0);
    num::Vector y(local_n, 2.0);

    // Computes global dot product across all ranks
    double global_dot = num::mpi::dot(x, y);

    num::mpi::finalize();
}
```

---

## Exercises

1. Write an MPI program where each process computes its rank squared, then use `MPI_Reduce` to find the sum of all squares.

2. Implement a parallel vector addition \f$\mathbf{z} = \mathbf{x} + \mathbf{y}\f$ where vectors are distributed across processes.

3. Analyze the communication cost of a distributed matrix-vector product with an \f$n \times n\f$ matrix on \f$p\f$ processes (1D row distribution).

4. Modify the dot product to use non-blocking `MPI_Iallreduce`. What computation could you overlap?

5. What happens if you call `MPI_Recv` with `source = MPI_ANY_SOURCE`? When is this useful?

---

## References

- Gropp, Lusk & Skjellum, *Using MPI*
- MPI Forum, *MPI Standard* (mpi-forum.org)
- Pacheco, *Parallel Programming with MPI*
