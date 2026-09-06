# Containers {#page_container}

## Vectors

### Construct a vector

```cpp
num::vec empty;             // size() == 0
num::vec zeros(3);          // {0, 0, 0}
num::vec filled(3, 2.0);    // {2, 2, 2}
num::vec values{1.0, 2.0};  // {1, 2}
```

Construct from existing contiguous values:

```cpp
std::array<num::real, 3> source{1.0, 2.0, 3.0};
num::vec x(std::span<const num::real>(source)); // Copy source.
```

### Read and write entries

```cpp
num::vec x{1.0, 2.0, 3.0};
x[1] = 8.0;             // x is now {1, 8, 3}.
num::real value = x[2]; // value == 3.
num::idx count = x.size();
```

### Iterate over entries

```cpp
num::vec x{1.0, 2.0, 3.0};
for (num::real& value : x) {
    value *= 2.0; // x becomes {2, 4, 6}.
}
```

### Copy into standard storage

```cpp
num::vec x{1.0, 2.0, 3.0};
std::vector<num::real> output(x.size());
num::copy_to(x, output); // output receives {1, 2, 3}.
```

```cpp
std::vector<num::real> too_small(2);
num::copy_to(x, too_small); // Throws: destination size does not match.
```

### Scale in place

```cpp
num::vec x{1.0, 2.0, 3.0};
num::scale(x, 0.5); // x becomes {0.5, 1, 1.5}.
```

### Add vectors

```cpp
num::vec x{1.0, 2.0};
num::vec y{3.0, 4.0};
num::vec sum(2, 0.0);

num::add(x, y, sum); // sum <- x+y == {4, 6}.
```

### Add a scaled vector

```cpp
num::vec x{1.0, 2.0};
num::vec y{4.0, 5.0};

num::axpy(-2.0, x, y); // y <- y-2*x == {2, 1}.
```

### Dot product and norm

```cpp
num::vec x{3.0, 4.0};
num::vec y{2.0, 1.0};

num::real product = num::dot(x, y); // 10
num::real length = num::norm(x);    // 5
```

### Operate on non-owning spans

```cpp
std::array<num::real, 3> x{1.0, 2.0, 3.0};
std::array<num::real, 3> y{4.0, 5.0, 6.0};

num::real product = num::dot(
    std::span<const num::real>(x),
    std::span<const num::real>(y)); // No vec allocation.
```

### View interleaved coordinates

```cpp
num::vec storage{1.0, 2.0, 3.0, 4.0}; // (1,2), (3,4)
num::vec2_view points{storage};

points.x(1) = 8.0; // storage becomes {1, 2, 8, 4}.
points.y(0) = 9.0; // storage becomes {1, 9, 8, 4}.
```

Use `Vec2ConstView` when the underlying vector is read-only.

## Dense Matrices

### Construct a matrix

```cpp
num::mat empty;
num::mat zeros(2, 3);       // Zero-initialized 2-by-3 matrix.
num::mat filled(2, 3, 1.5); // Every entry starts at 1.5.
```

### Read dimensions and entries

```cpp
num::mat A(2, 3, 0.0);
A(1, 2) = 7.0;

num::idx rows = A.rows(); // 2
num::idx cols = A.cols(); // 3
num::idx size = A.size(); // 6
num::real value = A(1, 2);
```

mat storage is contiguous and row-major:

```cpp
num::real* data = A.data();
data[(1 * A.cols()) + 2] = 9.0; // Same entry as A(1, 2).
```

### Matrix-vector multiplication

```cpp
num::mat A(2, 2, 0.0);
A(0, 0) = 2.0;
A(1, 1) = 3.0;

num::vec x{4.0, 5.0};
num::vec y(2, 0.0);
num::matvec(A, x, y); // y <- A*x == {8, 15}.
```

### Matrix-matrix multiplication

```cpp
num::mat C(A.rows(), A.rows(), 0.0);
num::matmul(A, A, C); // C <- A*A.
```

Choose a backend explicitly when needed (see @ref page_parallel for the full list):

```cpp
num::seq::matmul(A, A, C);  // Portable reference, forced.
num::matmul(A, A, C);       // num::accel: best backend the build detected.
```

### Add scaled matrices

```cpp
num::mat C(A.rows(), A.cols(), 0.0);
num::matadd(2.0, A, -1.0, B, C); // C <- 2*A-B.
```

### Select a matrix multiplication kernel

There is nothing to select. `num::matmul` dispatches to the configured backend
(`num::blas`, `num::omp`, `num::cuda`), and the portable `num::kernel::gemm`
underneath it sizes its own register tile and cache panel from the target, so
there is no block size or tile width for a caller to pass. Call a backend by
name to pin one:

```cpp
num::matmul(A, B, C);      // Configured backend.
num::seq::matmul(A, B, C); // Portable kernel, single-threaded.
num::omp::matmul(A, B, C); // Threaded, same kernel per tile.
```

## Matrix Construction Helpers

### Unit vector

```cpp
num::vec e = num::unit_vector(4, 2); // {0, 0, 1, 0}
```

```cpp
num::unit_vector(4, 4); // Throws: index is outside [0,4).
```

### Identity matrix

```cpp
num::mat I = num::identity(3); // 3-by-3 identity.
```

### Selected identity columns

```cpp
num::mat E = num::identity_columns(5, 1, 2);
// E contains columns 1 and 2 of the 5-by-5 identity.
```

### Read and write a diagonal

```cpp
num::vec diagonal = num::diagonal(A); // Copy A's main diagonal.
num::mat D = num::diagonal_matrix(
    std::span<const num::real>(diagonal.data(), diagonal.size()));
```

```cpp
std::array<num::real, 2> values{4.0, 5.0};
num::set_diagonal(A, values); // Replace A(0,0) and A(1,1).
```

### Transpose a matrix

```cpp
num::mat At = num::transpose(A); // At(j,i) == A(i,j).
```

## Element and Row Scaling

### Scale vector elements

```cpp
num::vec x{2.0, 3.0, 4.0};
std::array<num::real, 3> weights{1.0, 2.0, 0.5};

num::scale_elements(x, weights);  // x becomes {2, 6, 2}.
num::divide_elements(x, weights); // x returns to {2, 3, 4}.
```

### Scale matrix rows

```cpp
num::mat A(2, 2, 1.0);
std::array<num::real, 2> weights{2.0, 3.0};

num::scale_rows(A, weights);  // Rows become {2,2} and {3,3}.
num::divide_rows(A, weights); // Restore the original matrix.
```

Weight counts must match the vector size or matrix row count.

## Gather and Scatter

### Gather selected values

```cpp
std::array<num::real, 4> input{10.0, 20.0, 30.0, 40.0};
std::array<num::idx, 2> indices{3, 1};

auto selected = num::gather<num::real>(input, indices); // {40, 20}
```

### Scatter selected values

```cpp
std::array<num::real, 2> values{7.0, 8.0};
std::array<num::idx, 2> indices{2, 0};
std::array<num::real, 3> output{0.0, 0.0, 0.0};

num::scatter<num::real>(values, indices, output); // {8, 0, 7}
```

Accumulate instead of replacing:

```cpp
num::scatter<num::real>(values, indices, output, true); // Add into output.
```

## Sparse Matrices

### Construct from triplets

```cpp
auto A = num::spmat::from_triplets(
    3, 3,
    std::vector<num::idx>{0, 0, 1, 2},
    std::vector<num::idx>{0, 1, 1, 2},
    std::vector<num::real>{2.0, 1.0, 3.0, 4.0});
// Duplicate triplets are summed and stored as CSR.
```

### Multiply by a vector

```cpp
num::vec x{1.0, 2.0, 3.0};
num::vec y(3, 0.0);
num::sparse_matvec(A, x, y); // y <- A*x.
```

### Transform sparse storage

```cpp
num::spmat At = num::transpose(A);
num::spmat half = num::scaled(A, 0.5);
num::mat dense_A = num::dense(A);
num::vec diagonal = num::diagonal(A);
```

### Apply a diagonal similarity transform

```cpp
std::array<num::real, 3> weights{1.0, 2.0, 4.0};
num::mat transformed = num::diagonal_similarity(A, weights);
// transformed == D^-1*A*D.
```

## Matrix Properties

### Check a property

```cpp
bool symmetric = num::linear::is_symmetric(A_dense);
bool positive_definite = num::linear::is_spd(A_dense);
```

### Validate and wrap a property

```cpp
num::mat A_dense = num::identity(3);

auto symmetric = num::linear::make_symmetric(A_dense); // Checks A==A^T.
auto spd = num::linear::make_spd(A_dense);             // Checks A==A^T>0.
```

```cpp
num::mat indefinite(2, 2, 0.0);
indefinite(0, 0) = 1.0;
indefinite(1, 1) = -1.0;

auto spd = num::linear::make_spd(indefinite); // Throws when validation fails.
```

### Declare a construction-guaranteed property

```cpp
num::mat A_dense = num::identity(3);

auto symmetric = num::linear::assume_symmetric(A_dense);
auto spd = num::linear::assume_spd(A_dense);
// No numerical validation is performed.
```

## Concepts and Runtime Diagnostics

### Check storage interfaces at compile time

```cpp
static_assert(num::vector_space<num::vec>);
static_assert(num::mutable_vector_space<num::vec>);
static_assert(num::repr::contiguous<num::vec>);
static_assert(num::matrix_space<num::mat>);
```

Concepts inspect available operations and property tags. They do not inspect
the numerical values stored in an object.

### Check runtime dimensions and values

```cpp
num::debug::check_dim(A.rows(), x.size(), "A*x"); // Throws on a size mismatch.
num::debug::check_non_empty(x.size(), "x");       // Throws for an empty vector.
num::debug::check_finite(x.data(), x.size(), "x"); // Throws on NaN or infinity.
```

### Select the diagnostic level

```cpp
num::debug::set_level(num::debug::diagnostic_level::full);  // Basic and property checks.
num::debug::set_level(num::debug::diagnostic_level::basic); // Dimensions and values only.
num::debug::set_level(num::debug::diagnostic_level::off);   // Skip debug checks.
```

### Validate an operator before tagging it

```cpp
num::mat A_dense = num::identity(3);

num::operators::dense_op op(A_dense);
static_assert(num::linear_operator<decltype(op)>);

auto spd = num::operators::assume_spd(op); // Sample x^T*A*x when diagnostics are full.
static_assert(num::spd_operator<decltype(spd)>);
```

`assume_symmetric` similarly samples `x^T*A*y` against `y^T*A*x`. Once the wrapper
adds the property tag, constrained solvers can reject incompatible operators at
compile time.

## Selection and Probability

### Find the first maximum

```cpp
std::array<num::real, 4> values{1.0, 4.0, 4.0, 2.0};
num::idx best = num::argmax(std::span<const num::real>(values)); // 1
```

### Maximize a projected score

```cpp
num::idx best = num::argmax(values.size(), [&](num::idx index) {
    return -std::abs(values[index]); // Select the smallest absolute value.
});
```

### Select the smallest entries

```cpp
auto indices = num::smallest_indices(
    std::span<const num::real>(values), 2); // Indices sorted by value.
```

### Normalize nonnegative mass

```cpp
std::array<num::real, 3> probability{0.2, -0.1, 0.8};
num::real mass = num::clip_and_normalize_nonnegative(probability);
// Negative entries become zero; the result sums to one.
```

### Compute a weighted projection

```cpp
std::array<num::real, 3> probability{0.2, 0.3, 0.5};
num::real mean = num::weighted_sum(probability, [](num::idx state) {
    return static_cast<num::real>(state);
});
```

## Complete Program

@example 00_core_storage_and_helpers.cpp
