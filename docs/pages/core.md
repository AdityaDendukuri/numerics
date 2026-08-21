# Core Storage and Helper Examples {#page_core}

The core API uses owning `Vector` and row-major `Matrix` containers. Most
operations write into caller-provided output so allocations remain explicit.

## Vector Construction and Arithmetic

```cpp
num::Vector x{1.0, 2.0, 3.0}; // Copy an initializer list.
num::Vector y(3, 2.0);        // Fill three entries with 2.
num::Vector z(3, 0.0);        // Allocate reusable output.

num::scale(y, 0.5);    // y <- 0.5 y
num::add(x, y, z);     // z <- x+y
num::axpy(-1.0, x, z); // z <- z-x

double xy = num::dot(x, y);
double nx = num::norm(x);
```

Use spans when data is already owned elsewhere:

```cpp
std::span<const num::real> xs(x.data(), x.size());
std::span<const num::real> ys(y.data(), y.size());
double xy = num::dot(xs, ys);

std::vector<num::real> host(x.size());
num::copy_to(x, host);
```

Interleaved particle coordinates can be viewed without copying:

```cpp
num::Vector storage{1.0, 2.0, 3.0, 4.0};
num::Vec2View points{storage};
points.x(1) = 5.0; // storage[2] is now 5.
```

## Matrix Arithmetic

```cpp
num::Matrix A(3, 3, 0.0);
A(0, 0) = 4.0;
A(1, 1) = 5.0;
A(2, 2) = 6.0;

num::Vector y(3, 0.0);
num::matvec(A, x, y); // y <- A*x

num::Matrix At = num::transpose(A);
num::Matrix C(3, 3, 0.0);
num::matmul(A, At, C); // C <- A*At

num::Matrix S(3, 3, 0.0);
num::matadd(1.0, A, 1.0, At, S); // S <- A+At
```

`matmul_blocked`, `matmul_register_blocked`, and `matmul_simd` expose specific
CPU kernels. Regular `matmul` uses the selected runtime backend.

## Matrix Construction and Scaling

```cpp
auto e2 = num::unit_vector(3, 1);       // [0,1,0]
auto I = num::identity(3);              // 3-by-3 identity
auto E = num::identity_columns(3, 1, 2); // Columns 1 and 2 of I

auto d = num::diagonal(A);
auto D = num::diagonal_matrix(
    std::span<const num::real>(d.data(), d.size()));
num::set_diagonal(A, std::array<num::real, 3>{2.0, 3.0, 4.0});

std::array<num::real, 3> weights{1.0, 2.0, 4.0};
num::scale_elements(x, weights);
num::divide_elements(x, weights);
num::scale_rows(A, weights);
num::divide_rows(A, weights);
```

## Gather, Scatter, and Selection

```cpp
std::vector<double> input{10.0, 20.0, 30.0};
std::array<num::idx, 2> indices{2, 0};

auto selected = num::gather<double>(input, indices); // [30,10]
std::vector<double> output(3, 0.0);
num::scatter<double>(selected, indices, output);

num::idx largest = num::argmax(std::span<const double>(input));
auto two_smallest = num::smallest_indices(
    std::span<const double>(input), 2);
```

For a projected score, avoid creating a temporary vector:

```cpp
num::idx least_absolute = num::argmax(input.size(), [&](num::idx i) {
    return -std::abs(input[i]);
});
```

## Sparse Matrices

```cpp
auto A = num::SparseMatrix::from_triplets(
    3, 3,
    std::vector<num::idx>{0, 0, 1, 2},
    std::vector<num::idx>{0, 1, 1, 2},
    std::vector<num::real>{2.0, 1.0, 3.0, 4.0});

num::Vector y(3, 0.0);
num::sparse_matvec(A, x, y);

auto At = num::transpose(A);
auto half = num::scaled(A, 0.5);
auto dense_A = num::dense(A);
auto diagonal = num::diagonal(A);
```

`from_csc` imports zero-based compressed-column data and converts it to native
CSR storage. `diagonal_similarity(A, weights)` returns

\f[
    D^{-1} A D, \qquad D=\operatorname{diag}(\text{weights}).
\f]

## Matrix Properties

```cpp
bool symmetric = num::linalg::is_symmetric(A);
bool spd = num::linalg::is_spd(A);

auto checked = num::linalg::make_spd(A);  // Validate, then wrap.
auto assumed = num::linalg::assume_spd(A); // Construction guarantees it.
```

Checked wrappers throw when the requested property is not satisfied. Assumed
wrappers express an invariant already guaranteed by matrix construction.

## Probability Helpers

```cpp
num::Vector p{0.2, -0.1, 0.8};
double mass = num::clip_and_normalize_nonnegative(
    std::span<num::real>(p.data(), p.size()));

double mean = num::weighted_sum(
    std::span<const num::real>(p.data(), p.size()),
    [](num::idx state) { return static_cast<double>(state); });
```

The complete demonstration is compiled as part of the examples build:
@example 00_core_storage_and_helpers.cpp
