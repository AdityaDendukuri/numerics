# Containers {#page_container}

## Vectors

### Construct a vector

```cpp
num::Vector empty;             // size() == 0
num::Vector zeros(3);          // {0, 0, 0}
num::Vector filled(3, 2.0);    // {2, 2, 2}
num::Vector values{1.0, 2.0};  // {1, 2}
```

Construct from existing contiguous values:

```cpp
std::array<num::real, 3> source{1.0, 2.0, 3.0};
num::Vector x(std::span<const num::real>(source)); // Copy source.
```

### Read and write entries

```cpp
num::Vector x{1.0, 2.0, 3.0};
x[1] = 8.0;             // x is now {1, 8, 3}.
num::real value = x[2]; // value == 3.
num::idx count = x.size();
```

### Iterate over entries

```cpp
num::Vector x{1.0, 2.0, 3.0};
for (num::real& value : x) {
    value *= 2.0; // x becomes {2, 4, 6}.
}
```

### Copy into standard storage

```cpp
num::Vector x{1.0, 2.0, 3.0};
std::vector<num::real> output(x.size());
num::copy_to(x, output); // output receives {1, 2, 3}.
```

```cpp
std::vector<num::real> too_small(2);
num::copy_to(x, too_small); // Throws: destination size does not match.
```

### Scale in place

```cpp
num::Vector x{1.0, 2.0, 3.0};
num::scale(x, 0.5); // x becomes {0.5, 1, 1.5}.
```

### Add vectors

```cpp
num::Vector x{1.0, 2.0};
num::Vector y{3.0, 4.0};
num::Vector sum(2, 0.0);

num::add(x, y, sum); // sum <- x+y == {4, 6}.
```

### Add a scaled vector

```cpp
num::Vector x{1.0, 2.0};
num::Vector y{4.0, 5.0};

num::axpy(-2.0, x, y); // y <- y-2*x == {2, 1}.
```

### Dot product and norm

```cpp
num::Vector x{3.0, 4.0};
num::Vector y{2.0, 1.0};

num::real product = num::dot(x, y); // 10
num::real length = num::norm(x);    // 5
```

### Operate on non-owning spans

```cpp
std::array<num::real, 3> x{1.0, 2.0, 3.0};
std::array<num::real, 3> y{4.0, 5.0, 6.0};

num::real product = num::dot(
    std::span<const num::real>(x),
    std::span<const num::real>(y)); // No Vector allocation.
```

### View interleaved coordinates

```cpp
num::Vector storage{1.0, 2.0, 3.0, 4.0}; // (1,2), (3,4)
num::Vec2View points{storage};

points.x(1) = 8.0; // storage becomes {1, 2, 8, 4}.
points.y(0) = 9.0; // storage becomes {1, 9, 8, 4}.
```

Use `Vec2ConstView` when the underlying vector is read-only.

## Dense Matrices

### Construct a matrix

```cpp
num::Matrix empty;
num::Matrix zeros(2, 3);       // Zero-initialized 2-by-3 matrix.
num::Matrix filled(2, 3, 1.5); // Every entry starts at 1.5.
```

### Read dimensions and entries

```cpp
num::Matrix A(2, 3, 0.0);
A(1, 2) = 7.0;

num::idx rows = A.rows(); // 2
num::idx cols = A.cols(); // 3
num::idx size = A.size(); // 6
num::real value = A(1, 2);
```

Matrix storage is contiguous and row-major:

```cpp
num::real* data = A.data();
data[(1 * A.cols()) + 2] = 9.0; // Same entry as A(1, 2).
```

### Matrix-vector multiplication

```cpp
num::Matrix A(2, 2, 0.0);
A(0, 0) = 2.0;
A(1, 1) = 3.0;

num::Vector x{4.0, 5.0};
num::Vector y(2, 0.0);
num::matvec(A, x, y); // y <- A*x == {8, 15}.
```

### Matrix-matrix multiplication

```cpp
num::Matrix C(A.rows(), A.rows(), 0.0);
num::matmul(A, A, C); // C <- A*A.
```

Choose a backend explicitly when needed:

```cpp
num::matmul(A, A, C, num::backend::seq);  // Sequential backend.
num::matmul(A, A, C, num::backend::dflt);  // Best compiled backend.
```

### Add scaled matrices

```cpp
num::Matrix C(A.rows(), A.cols(), 0.0);
num::matadd(2.0, A, -1.0, B, C); // C <- 2*A-B.
```

### Select a matrix multiplication kernel

```cpp
num::matmul_blocked(A, B, C, 64);             // Cache-blocked.
num::matmul_register_blocked(A, B, C, 64, 4); // Cache and register blocked.
num::matmul_simd(A, B, C, 64);                // AVX, NEON, or fallback.
```

Regular `matmul` dispatches through the selected runtime backend.

## Matrix Construction Helpers

### Unit vector

```cpp
num::Vector e = num::unit_vector(4, 2); // {0, 0, 1, 0}
```

```cpp
num::unit_vector(4, 4); // Throws: index is outside [0,4).
```

### Identity matrix

```cpp
num::Matrix I = num::identity(3); // 3-by-3 identity.
```

### Selected identity columns

```cpp
num::Matrix E = num::identity_columns(5, 1, 2);
// E contains columns 1 and 2 of the 5-by-5 identity.
```

### Read and write a diagonal

```cpp
num::Vector diagonal = num::diagonal(A); // Copy A's main diagonal.
num::Matrix D = num::diagonal_matrix(
    std::span<const num::real>(diagonal.data(), diagonal.size()));
```

```cpp
std::array<num::real, 2> values{4.0, 5.0};
num::set_diagonal(A, values); // Replace A(0,0) and A(1,1).
```

### Transpose a matrix

```cpp
num::Matrix At = num::transpose(A); // At(j,i) == A(i,j).
```

## Element and Row Scaling

### Scale vector elements

```cpp
num::Vector x{2.0, 3.0, 4.0};
std::array<num::real, 3> weights{1.0, 2.0, 0.5};

num::scale_elements(x, weights);  // x becomes {2, 6, 2}.
num::divide_elements(x, weights); // x returns to {2, 3, 4}.
```

### Scale matrix rows

```cpp
num::Matrix A(2, 2, 1.0);
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
auto A = num::SparseMatrix::from_triplets(
    3, 3,
    std::vector<num::idx>{0, 0, 1, 2},
    std::vector<num::idx>{0, 1, 1, 2},
    std::vector<num::real>{2.0, 1.0, 3.0, 4.0});
// Duplicate triplets are summed and stored as CSR.
```

### Multiply by a vector

```cpp
num::Vector x{1.0, 2.0, 3.0};
num::Vector y(3, 0.0);
num::sparse_matvec(A, x, y); // y <- A*x.
```

### Transform sparse storage

```cpp
num::SparseMatrix At = num::transpose(A);
num::SparseMatrix half = num::scaled(A, 0.5);
num::Matrix dense_A = num::dense(A);
num::Vector diagonal = num::diagonal(A);
```

### Apply a diagonal similarity transform

```cpp
std::array<num::real, 3> weights{1.0, 2.0, 4.0};
num::Matrix transformed = num::diagonal_similarity(A, weights);
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
auto symmetric = num::linear::make_symmetric(A_dense); // Checks A==A^T.
auto spd = num::linear::make_spd(A_dense);             // Checks A==A^T>0.
```

```cpp
auto spd = num::linear::make_spd(indefinite); // Throws when validation fails.
```

### Declare a construction-guaranteed property

```cpp
auto symmetric = num::linear::assume_symmetric(A_dense);
auto spd = num::linear::assume_spd(A_dense);
// No numerical validation is performed.
```

## Concepts and Runtime Diagnostics

### Check storage interfaces at compile time

```cpp
static_assert(num::VectorSpace<num::Vector>);
static_assert(num::MutableVectorSpace<num::Vector>);
static_assert(num::repr::Contiguous<num::Vector>);
static_assert(num::MatrixSpace<num::Matrix>);
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
num::debug::set_level(num::debug::DiagnosticLevel::full);  // Basic and property checks.
num::debug::set_level(num::debug::DiagnosticLevel::basic); // Dimensions and values only.
num::debug::set_level(num::debug::DiagnosticLevel::off);   // Skip debug checks.
```

### Validate an operator before tagging it

```cpp
num::operators::DenseOp op(A_dense);
static_assert(num::LinearOperator<decltype(op)>);

auto spd = num::operators::assume_spd(op); // Sample x^T*A*x when diagnostics are full.
static_assert(num::SPDOperator<decltype(spd)>);
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
