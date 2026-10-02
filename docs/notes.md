# Algorithm notes

Derivations, measurements and design reasons behind specific routines. The reference pages link here where a header comment is too short to hold them.

## Container vocabulary

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

**Containers.** Every standard container is used through a `num::` alias, so a
declaration never mixes `std::` and `num::`.
Two sequence containers are named after what they hold, because their C++ names are words
this library has already spent on mathematics.

| alias | is | the word it frees |
|---|---|---|
| `num::array<T>` | `std::vector<T>` | `num::vec<num::real>`, an element of a vector space |
| `num::static_array<T, N>` | `std::array<T, N>` | — |
| `num::view<T>` | `std::span<T>` | the span of a set of vectors |

The associative containers keep their C++ names.
A plain name keeps its keys sorted, and an `unordered_` name hashes them.

| alias | order | C++ | Rust |
|---|---|---|---|
| `num::unordered_map<K, V>` | none | `std::unordered_map<K, V>` | `std::collections::HashMap<K, V>` |
| `num::map<K, V>` | sorted | `std::map<K, V>` | `std::collections::BTreeMap<K, V>` |
| `num::unordered_set<K>` | none | `std::unordered_set<K>` | `std::collections::HashSet<K>` |
| `num::set<K>` | sorted | `std::set<K>` | `std::collections::BTreeSet<K>` |

A declaration then says which half of the library it belongs to:

```cpp
num::array<num::idx> row_offsets;   // storage
num::vec<num::real>             x(4);          // mathematics
```

One member function has an alias as well.
`num::append(values, x)` is `values.emplace_back(x)`, named for what the caller does rather than for how the element is stored, and it takes either a value or the arguments of the element's constructor.
An alias cannot be a member of a type alias, so it is a free function; `push_back` remains available and means the same thing.

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

### Choosing between num::vec<num::real> and num::array<num::real>

Both hold `double`s contiguously, and they are different types with different jobs.

```cpp
num::array<num::real> raw(n);   // storage: grows, zero-initialises, 16-byte aligned
num::vec<num::real>              x(n);     // mathematics: fixed extent, 64-byte aligned
```

`num::vec<num::real>` owns over-aligned storage, skips the zero-initialising pass when the contents
are about to be overwritten, and satisfies `num::math::vector_space`, so solvers and
operators take it directly. It has no `push_back`: its extent is fixed at construction.
Use `num::array` when the length is not known until the values are, and `num::vec<num::real>` once
the data is mathematics.

## Over-aligned storage

The dense containers allocate through `num::make_aligned`, which places the first element on
a `num::storage_alignment` boundary (64 bytes by default). Plain `new T[n]` guarantees only
16.

On AArch64/NEON and x86-64/AVX-512, clang emits identical code for the level-1 kernels with
and without the guarantee. It already prefers the unaligned move forms, which cost nothing on
an aligned address. The `std::assume_aligned` in `data()` is there for kernels that will need
it, not for speed today.

The alignment buys three structural properties.

- A vector load never straddles a cache line.
- Two containers never share a line at their boundaries, so a threaded reduction over
  adjacent buffers has no false sharing.
- Aligned SIMD loads, non-temporal stores and pinned host memory registration require it.

It also costs something. The aligned `operator new` bypasses the small-size fast path of
libc++, which on macOS measured about 4x the latency of plain `new` below a kilobyte. Only
code that allocates inside a loop sees this.

The guarantee covers the base pointer. `A.data() + j` is aligned only when `j * sizeof(T)`
is a multiple of `storage_alignment`, so a matrix row starts on a boundary only when its
stride does. The stride is not padded.

## Spanning-tree clique sampling

`structures/graph/clique.hpp` replaces the fill of one elimination by a random spanning tree
of it, reweighted so the elimination stays unbiased in expectation.

### The symmetric clique

Eliminating `v` adds a clique on its neighbours with Schur-complement weights
$w_{ij} = c_i c_j / C$, where $c_i$ is the conductance from `v` to neighbour `i`
and $C = \sum_i c_i$. For a weighted uniform spanning tree, Kirchhoff gives the inclusion
probability $p_e = w_e R_{\mathrm{eff}}(e)$. Effective resistances usually need
Laplacian solves, but this clique Laplacian is $\operatorname{diag}(c) - cc^{T}/C$, and
solving it gives the series path through `v`:
$$
  R_{\mathrm{eff}}(i,j) = \frac{1}{c_i} + \frac{1}{c_j}, \qquad
  p_{ij} = \frac{c_i + c_j}{C}, \qquad
  \sum_{i<j} p_{ij} = d - 1.
$$
The sum is the edge count of a spanning tree. The reweighting
$\widehat c_{ij} = w_{ij}/p_{ij} = c_i c_j / (c_i + c_j)$ is the harmonic mean, the same
series conductance the independent sampler assigns, so
$\mathbb{E}[\widetilde L^{(v)}] = \mathrm{Sc}(L)$.

Aldous--Broder degenerates on this clique. Its transition kernel
$P(i \to j) = c_j / (C - c_i)$ does not depend on the current vertex, so the walk is a
coupon collector: draw i.i.d. from $c/C$ and record the entering edge at each first visit.

### The directed biclique

A nonsymmetric pivot leaves the rank-one update $xy^{T}/a$, with $x_i = -A_{iv}$,
$y_j = -A_{vj}$ and $a = A_{vv}$. Lifting to a bipartite graph with vertices
$i_L$ for in-neighbours, $j_R$ for out-neighbours, and conductances $x_iy_j/a$
keeps the two directions of an edge distinct. With $X = \sum_i x_i$ and
$Y = \sum_j y_j$,
$$
  R_{\mathrm{eff}}(i_L, j_R) = a\Big[\tfrac{1}{x_iY} + \tfrac{1}{y_jX} - \tfrac{1}{XY}\Big],
  \qquad
  p_{ij} = 1 - (1-\alpha_i)(1-\beta_j),
$$
where $\alpha_i = x_i/X$ and $\beta_j = y_j/Y$. The probabilities sum to $m + n - 1$,
the edge count of a spanning tree on the duplicated vertices. The walk alternates the fixed
distributions $y/Y$ and $x/X$. The conductance reading needs $x_i, y_j, a > 0$.
Matrix Chernoff bounds for strongly Rayleigh measures are Hermitian results, so they say
nothing about concentration for this nonsymmetric update.

### Measured behaviour

On 2D grid Laplacians with n up to 40k, one tree per elimination is worse than the
independent sampler: 44 PCG iterations against 41 for `ac1` and 29 for `ac2`. On
higher-degree graphs with skewed weights the coupon collector also makes setup several
times more expensive.

Averaging k trees per elimination concentrates, as the strongly Rayleigh Chernoff theory
predicts. On a 14400-vertex grid it takes 44, 27, 20 and 15 iterations at k = 1, 2, 4 and 8,
against 29 for `ac2`. Setup grows faster than k, because each tree raises the degree of later
eliminations: 11 ms, 82 ms, 616 ms and 2.7 s. It pays only when one factorization serves many
solves, about 50 to break even at k = 2 and 600 at k = 8.

## Chebyshev polynomial preconditioning

`num::chebyshev_preconditioner` approximates $A^{-1}$ by the degree-m polynomial that
minimizes $\max_{\lambda \in [\ell, h]} |1 - \lambda p(\lambda)|$. It needs only the
operator's action, so it is the one preconditioner here that works on a matrix-free operator
built with `num::operators::make_op`. It also uses no global reductions.

A degree-m polynomial improves the condition number by about a factor of m. That is not
competitive on a second-order elliptic operator, whose condition number grows with the mesh;
use ApproxChol for SDD systems, or algebraic multigrid. It fits three cases:

- matrix-free operators, which have no entries to factor;
- moderately conditioned systems, such as mass matrices, shifted or damped operators, and
  regularized least squares;
- smoothing the upper spectrum inside a multigrid cycle.

The polynomial is positive definite only when $0 < \ell \le \lambda_{\min}$ and
$\lambda_{\max} \le h$. Bounds that miss part of the spectrum make it indefinite, and PCG
then loses its monotone error. `num::estimate_largest_eigenvalue` finds $h$ by power
iteration. There is no equally cheap estimate of $\ell$. Take it from the problem's
physics, from a shift or regularization parameter, or from `num::lanczos`, not from an assumed
condition number.

## Probed inverse diagonal of an M-matrix

`num::inverse_diagonal(factor, A, symmetrizer, options)` estimates $\operatorname{diag}(A^{-1})$
for a sparse nonsingular M-matrix from one block of Gaussian probes, instead of one solve per
entry.

Two solves give $q = A^{-1}\mathbf{1}$ and $r = A^{-T}\mathbf{1}$, both positive. With
$D = \operatorname{Diag}(\sqrt{r_j/q_j})$, the symmetric part $S$ of
$\tilde{A} = DAD^{-1}$ is positive definite. A diagonal similarity leaves the inverse
diagonal unchanged, so $\operatorname{diag}(\tilde{A}^{-1}S\tilde{A}^{-T}) = \operatorname{diag}(A^{-1})$.
An approximate Cholesky factorization $S = \tilde{K}\tilde{S}\tilde{K}^{T}$ makes this a Gram
matrix. Entry j is then $\|\delta_j^{T}\tilde{A}^{-1}\tilde{K}\tilde{S}^{1/2}\|_2^2$, and the
row-wise mean square over Gaussian probes estimates it without bias.

Each row of the probed block is chi-square with `probes` degrees of freedom. The estimate
therefore concentrates multiplicatively and stays positive.

When A is similar to a symmetric matrix through $\operatorname{Diag}(\sqrt{\pi})$, passing
$\sqrt{\pi}$ as `symmetrizer` replaces the scaling above. That needs no solves, and the
square root is applied in inverse form.
