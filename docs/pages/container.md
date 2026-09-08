# Containers {#page_container}

## All containers

Grouped by what each does. Types own their storage and are
over-aligned; the free functions write into caller-provided destinations.

<div class="sym-index">
<div class="kidx-group"><span class="kidx-title">Scalars and vocabulary <span class="hdr">&lt;core/types.hpp&gt;</span></span><br/><span class="kidx-syms">@ref num::real "real" &ndash; @ref num::idx "idx" &ndash; @ref num::cplx "cplx" &ndash; @ref num::array "array" &ndash; @ref num::static_array "static_array" &ndash; @ref num::view "view" &ndash; @ref num::table "table" &ndash; @ref num::sorted_table "sorted_table" &ndash; @ref num::key_set "key_set"</span></div><div class="kidx-group"><span class="kidx-title">Dense vectors <span class="hdr">&lt;container/vector.hpp&gt;</span></span><br/><span class="kidx-syms">@ref num::basic_vec "basic_vec" &ndash; @ref num::vec "vec" &ndash; @ref num::cvec "cvec" &ndash; @ref num::vec2_view "vec2_view" &ndash; @ref num::copy_to "copy_to"</span></div><div class="kidx-group"><span class="kidx-title">Dense matrices <span class="hdr">&lt;container/matrix.hpp&gt;</span></span><br/><span class="kidx-syms">@ref num::basic_mat "basic_mat" &ndash; @ref num::mat "mat"</span></div><div class="kidx-group"><span class="kidx-title">Vector arithmetic <span class="hdr">&lt;container/vector_ops.hpp&gt;</span></span><br/><span class="kidx-syms">@ref num::scale "scale" &ndash; @ref num::axpy "axpy" &ndash; @ref num::axpby "axpby" &ndash; @ref num::axpbyz "axpbyz" &ndash; @ref num::add "add" &ndash; @ref num::dot "dot" &ndash; @ref num::norm "norm"</span></div><div class="kidx-group"><span class="kidx-title">Reductions <span class="hdr">&lt;container/reduce.hpp&gt;</span></span><br/><span class="kidx-syms">@ref num::sum "sum" &ndash; @ref num::l1_norm "l1_norm" &ndash; @ref num::linf_norm "linf_norm"</span></div><div class="kidx-group"><span class="kidx-title">Matrix arithmetic <span class="hdr">&lt;container/matrix_ops.hpp&gt;</span></span><br/><span class="kidx-syms">@ref num::matmul "matmul" &ndash; @ref num::matvec "matvec" &ndash; @ref num::matadd "matadd"</span></div><div class="kidx-group"><span class="kidx-title">Rank-1 and triangular solves <span class="hdr">&lt;container/dense.hpp&gt;</span></span><br/><span class="kidx-syms">@ref num::ger "ger" &ndash; @ref num::trsv_lower "trsv_lower" &ndash; @ref num::trsv_upper "trsv_upper"</span></div><div class="kidx-group"><span class="kidx-title">Matrix construction <span class="hdr">&lt;linear/matrix_utils.hpp&gt;</span></span><br/><span class="kidx-syms">@ref num::identity "identity" &ndash; @ref num::eye "eye" &ndash; @ref num::zeros "zeros" &ndash; @ref num::ones "ones" &ndash; @ref num::unit_vector "unit_vector" &ndash; @ref num::identity_columns "identity_columns" &ndash; @ref num::diagonal_matrix "diagonal_matrix" &ndash; @ref num::transpose "transpose"</span></div><div class="kidx-group"><span class="kidx-title">Diagonals <span class="hdr">&lt;linear/matrix_utils.hpp&gt;</span></span><br/><span class="kidx-syms">@ref num::diagonal "diagonal" &ndash; @ref num::set_diagonal "set_diagonal"</span></div><div class="kidx-group"><span class="kidx-title">Scaling and accumulation <span class="hdr">&lt;linear/matrix_utils.hpp&gt;</span></span><br/><span class="kidx-syms">@ref num::scale_elements "scale_elements" &ndash; @ref num::scale_rows "scale_rows" &ndash; @ref num::divide_elements "divide_elements" &ndash; @ref num::divide_rows "divide_rows" &ndash; @ref num::accu "accu"</span></div><div class="kidx-group"><span class="kidx-title">Gather and scatter <span class="hdr">&lt;linear/matrix_utils.hpp&gt;</span></span><br/><span class="kidx-syms">@ref num::gather "gather" &ndash; @ref num::scatter "scatter"</span></div><div class="kidx-group"><span class="kidx-title">Sparse matrices <span class="hdr">&lt;linear/sparse/sparse.hpp&gt;</span></span><br/><span class="kidx-syms">@ref num::spmat "spmat" &ndash; @ref num::spmat::from_triplets "spmat::from_triplets" &ndash; @ref num::spmat::from_csc "spmat::from_csc" &ndash; @ref num::sparse_matvec "sparse_matvec" &ndash; @ref num::dense "dense" &ndash; @ref num::spmat::nnz "spmat::nnz"</span></div><div class="kidx-group"><span class="kidx-title">Small fixed-size <span class="hdr">&lt;container/small_matrix.hpp&gt;</span></span><br/><span class="kidx-syms">@ref num::small_vec "small_vec" &ndash; @ref num::small_matrix "small_matrix" &ndash; @ref num::givens_rotation "givens_rotation"</span></div><div class="kidx-group"><span class="kidx-title">Discrete indices <span class="hdr">&lt;container/multi_index.hpp&gt;</span></span><br/><span class="kidx-syms">@ref num::multi_index "multi_index"</span></div><div class="kidx-group"><span class="kidx-title">Over-aligned storage <span class="hdr">&lt;container/util/aligned_storage.hpp&gt;</span></span><br/><span class="kidx-syms">@ref num::aligned_array "aligned_array" &ndash; @ref num::make_aligned "make_aligned" &ndash; @ref num::make_aligned_for_overwrite "make_aligned_for_overwrite" &ndash; @ref num::is_storage_aligned "is_storage_aligned"</span></div><div class="kidx-group"><span class="kidx-title">Orthogonal polynomials <span class="hdr">&lt;container/util/math.hpp&gt;</span></span><br/><span class="kidx-syms">@ref num::legendre "legendre" &ndash; @ref num::assoc_legendre "assoc_legendre" &ndash; @ref num::laguerre "laguerre" &ndash; @ref num::assoc_laguerre "assoc_laguerre" &ndash; @ref num::hermite "hermite"</span></div><div class="kidx-group"><span class="kidx-title">Bessel functions <span class="hdr">&lt;container/util/math.hpp&gt;</span></span><br/><span class="kidx-syms">@ref num::bessel_j "bessel_j" &ndash; @ref num::bessel_y "bessel_y" &ndash; @ref num::bessel_i "bessel_i" &ndash; @ref num::sph_bessel_j "sph_bessel_j" &ndash; @ref num::sph_bessel_y "sph_bessel_y"</span></div><div class="kidx-group"><span class="kidx-title">Ranges and sampling <span class="hdr">&lt;container/util/math.hpp&gt;</span></span><br/><span class="kidx-syms">@ref num::linspace "linspace" &ndash; @ref num::logspace "logspace" &ndash; @ref num::int_range "int_range" &ndash; @ref num::rng_state "rng_state" &ndash; @ref num::rng_uniform "rng_uniform" &ndash; @ref num::rng_normal "rng_normal" &ndash; @ref num::rng_int "rng_int"</span></div><div class="kidx-group"><span class="kidx-title">Misc numerics <span class="hdr">&lt;container/util/integer_pow.hpp&gt;</span></span><br/><span class="kidx-syms">@ref num::ipow "ipow"</span></div>
</div>

---

## 1. Vocabulary

Every type below is an alias, so it is the underlying type exactly — nothing is wrapped,
nothing converts, and a function expecting the standard type accepts the alias unchanged.

**Scalars.** These exist so precision and index width are decided in one place rather
than spelled out at every site.

| alias | is | use it for |
|---|---|---|
| `num::real` | `double` | every floating-point quantity |
| `num::idx` | `std::size_t` | sizes, offsets, and subscripts |
| `num::cplx` | `std::complex<num::real>` | complex amplitudes |

Note the spelling: `cplx`, not `cmplx`.

**Containers.** Four standard containers are named after words this library has already
spent on mathematics, so they get names that say what they hold instead.

| alias | is | the word it frees |
|---|---|---|
| `num::array<T>` | `std::vector<T>` | `num::vec`, an element of a vector space |
| `num::static_array<T, N>` | `std::array<T, N>` | — |
| `num::view<T>` | `std::span<T>` | the span of a set of vectors |
| `num::table<K, V>` | `std::unordered_map<K, V>` | a linear map |
| `num::sorted_table<K, V>` | `std::map<K, V>` | likewise |
| `num::key_set<K>` | `std::unordered_set<K>` | a set |

A declaration then says which half of the library it belongs to:

```cpp
num::array<num::idx> row_offsets;   // storage
num::vec             x(4);          // mathematics
```

Because these are aliases and not wrappers, standard code keeps working verbatim:

```cpp
num::array<num::real> a{1.0, 2.0, 3.0};
std::vector<num::real> &same = a;      // the same object; no conversion happens
std::sort(a.begin(), a.end());         // ordinary standard algorithms
```

Compiler diagnostics still name the underlying standard type. Containers whose names
carry no mathematical meaning — `std::pair`, `std::tuple`, `std::optional`,
`std::string` — are deliberately left alone, so the library does not end up maintaining a
parallel vocabulary for the whole standard library.

### Choosing between num::vec and num::array<num::real>

Both hold `double`s contiguously, and they are different types with different jobs.

```cpp
num::array<num::real> raw(n);   // storage: grows, zero-initialises, 16-byte aligned
num::vec              x(n);     // mathematics: fixed extent, 64-byte aligned
```

`num::vec` owns over-aligned storage, skips the zero-initialising pass when the contents
are about to be overwritten, and satisfies `num::math::vector_space`, so solvers and
operators take it directly. It has no `push_back`: its extent is fixed at construction.
Use `num::array` when the length is not known until the values are, and `num::vec` once
the data is mathematics.

---

## 2. Vectors

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
num::vec x(num::view<const num::real>(source)); // Copy source.
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
    num::view<const num::real>(x),
    num::view<const num::real>(y)); // No vec allocation.
```

### View interleaved coordinates

```cpp
num::vec storage{1.0, 2.0, 3.0, 4.0}; // (1,2), (3,4)
num::vec2_view points{storage};

points.x(1) = 8.0; // storage becomes {1, 2, 8, 4}.
points.y(0) = 9.0; // storage becomes {1, 9, 8, 4}.
```

Use `Vec2ConstView` when the underlying vector is read-only.

## 3. Dense Matrices

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

## 4. Matrix Construction Helpers

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
    num::view<const num::real>(diagonal.data(), diagonal.size()));
```

```cpp
std::array<num::real, 2> values{4.0, 5.0};
num::set_diagonal(A, values); // Replace A(0,0) and A(1,1).
```

### Transpose a matrix

```cpp
num::mat At = num::transpose(A); // At(j,i) == A(i,j).
```

## 5. Element and Row Scaling

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

## 6. Gather and Scatter

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

## 7. Sparse Matrices

### Construct from triplets

```cpp
auto A = num::spmat::from_triplets(
    3, 3,
    num::array<num::idx>{0, 0, 1, 2},
    num::array<num::idx>{0, 1, 1, 2},
    num::array<num::real>{2.0, 1.0, 3.0, 4.0});
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

## 8. Storage Layout and Foreign Types

The matrix operations are constrained on *layout*, not on `num::mat` and `num::spmat`. One
name serves every storage format, and the format decides the implementation:

| operation | dense row-major | compressed sparse row |
| :--- | :--- | :--- |
| `num::matvec(A, x, y)` | backend `gemv` | `kernel::spmv` |
| `num::transpose(A)` | → `num::mat` | → `num::spmat` |
| `num::diagonal(A)` | → `num::vec` | → `num::vec` |
| `num::dense(A)` | — | → `num::mat` |
| `num::scaled(A, a)` | → `num::mat` | → `num::spmat` |

The two concepts are disjoint — no type satisfies both — so the overloads never compete:

```cpp
num::mat   D = num::identity(3);
num::spmat S = num::spmat::from_triplets(3, 3, {0, 1, 2}, {0, 1, 2}, {2.0, 3.0, 4.0});
num::vec   x(3, 1.0), y(3, 0.0), z(3, 0.0);

num::matvec(D, x, y);   // selected by num::repr::dense_row_major
num::matvec(S, x, z);   // selected by num::repr::csr
```

### Using your own storage

Because the constraint is on the accessors rather than on a type, a matrix the library has
never heard of participates directly. No inheritance, no adapter, no trait specialisation
— this is the shape an Eigen sparse matrix or a raw triple of CSR buffers already has:

```cpp
struct my_csr {
    num::idx nr, nc;
    std::vector<num::idx> rp, ci;
    std::vector<num::real> vals;

    num::idx n_rows() const { return nr; }
    num::idx n_cols() const { return nc; }
    num::idx nnz()    const { return vals.size(); }
    const num::idx  *row_ptr() const { return rp.data(); }
    const num::idx  *col_idx() const { return ci.data(); }
    const num::real *values()  const { return vals.data(); }
};

static_assert(num::repr::csr<my_csr>);          // it already qualifies

// [[2 0 1], [0 3 0], [1 0 4]]
my_csr A{3, 3, {0, 2, 3, 5}, {0, 2, 1, 0, 2}, {2.0, 1.0, 3.0, 1.0, 4.0}};
num::vec ones(3, 1.0), out(3, 0.0);

num::matvec(A, ones, out);                      // row sums: 3, 3, 5
num::vec d  = num::diagonal(A);                 // 2, 3, 4
num::mat Ad = num::dense(A);                    // densified into a num::mat
```

A dense type qualifies on `rows()`, `cols()` and `data()`:

```cpp
struct my_dense {
    num::idx r, c;
    std::vector<num::real> v;

    num::idx rows() const { return r; }
    num::idx cols() const { return c; }
    const num::real *data() const { return v.data(); }
    num::real *data() { return v.data(); }
    num::real operator()(num::idx i, num::idx j) const { return v[(i * c) + j]; }
};

static_assert(num::repr::dense_row_major<my_dense>);

my_dense M{2, 2, {1.0, 2.0, 3.0, 4.0}};
num::vec u(2, 1.0), w(2, 0.0);
num::matvec(M, u, w);                           // 3, 7
num::mat Mt = num::transpose(M);
```

`num::mat` still reaches the configured backend — BLAS, OpenMP or CUDA — because the
dispatch checks for it. A foreign dense type takes the kernel path, which needs only the
three accessors the concept requires.

`num::sparse_matvec` remains as a forwarder to `num::matvec` for code that already calls it.

---

## 9. Matrix Properties

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

## 10. Concepts and Runtime Diagnostics

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

## 11. Selection and Probability

### Find the first maximum

```cpp
std::array<num::real, 4> values{1.0, 4.0, 4.0, 2.0};
num::idx best = num::argmax(num::view<const num::real>(values)); // 1
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
    num::view<const num::real>(values), 2); // Indices sorted by value.
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

## 12. Complete Program

@example 00_core_storage_and_helpers.cpp
