# Concepts, Invariants & Diagnostics {#page_concepts}

@note The dependency-free `core/math` protocol and immutable evidence API are
the canonical vocabulary for newly migrated generic algorithms. The property
wrappers and preset-controlled sampling described later on this page remain the
compatibility surface for domains that have not migrated yet. See @ref
page_refactor_roadmap "Mathematical Core Refactor Roadmap" for current status.

Numerical routines have preconditions. Cholesky needs a symmetric positive definite matrix, the conjugate gradient method needs the same, and Lanczos needs a symmetric one. Numerics states those preconditions in the type system so that violating one is a compile error rather than a wrong answer.

```cpp
#include <numerics.hpp>
```

---

## 1. Two Kinds of Requirement

Preconditions divide into two kinds, and the library treats them differently.

An **algebraic structure** is a property of a type. A vector space is a set with addition and scalar multiplication that is closed under both. Whether `num::Vector` is a vector space is decided by looking at `num::Vector`, so the compiler can answer it:

```cpp
static_assert(num::VectorSpace<num::Vector>);
static_assert(num::VectorSpace<std::vector<float>>);
static_assert(!num::VectorSpace<std::span<const double>>); // A view cannot hold a sum.
```

An **invariant** is a property of a particular value. Positive definiteness is a statement about the entries of one matrix:

\f[
\mathbf{x}^T A \mathbf{x} > 0 \quad \forall \mathbf{x} \neq 0
\f]

Two matrices of the same type can differ on it, so no amount of inspecting `num::Matrix` decides it. The compiler cannot answer this one.

Numerics handles the difference by making the caller state the invariant and recording that statement in the type:

```cpp
num::Matrix A = make_spd_matrix();

auto S = num::assume_spd(A);          // A claim, carried by the type of S.
static_assert(num::SPDMatrixLike<decltype(S)>);
```

`num::Matrix` and `decltype(S)` hold the same numbers. They differ in what has been claimed about them, and that difference is what a solver constrains on.

Because the claim is only a claim, it is checked at runtime. `num::assume_spd` samples the property, `num::make_spd` verifies it exhaustively, and `num::preset` controls how much of that work happens. Sections 2 through 5 cover each in turn.

---

## 2. Property Tags

Cholesky factorization is defined for symmetric positive definite matrices:

\f[
A = A^T, \qquad \mathbf{x}^T A \mathbf{x} > 0 \quad \forall \mathbf{x} \neq 0
\f]

Neither property can be read from a type. The caller attaches it:

```cpp
num::Matrix A = make_spd_matrix();

auto asserted = num::assume_spd(A); // Samples the property. O(n^2) under preset::strict.
auto verified = num::make_spd(A);   // Checks every entry in O(n^3). Throws if A is not SPD.

num::CholeskyResult factor = num::cholesky(asserted); // Computes A = L*L^T.
```

The tag travels with the value, so a factorization performed once can be reused without repeating the check.

### Available Taggers

| Tagger | Invariant | Cost |
| :--- | :--- | :--- |
| `num::assume_square(A)` | \f$\text{rows} = \text{cols}\f$ | None |
| `num::assume_symmetric(A)` | \f$A = A^T\f$ | Sampled |
| `num::assume_psd(A)` | \f$\mathbf{x}^T A \mathbf{x} \geq 0\f$ | Sampled |
| `num::assume_spd(A)` | \f$\mathbf{x}^T A \mathbf{x} > 0\f$ | Sampled |
| `num::assume_banded(A, kl, ku)` | \f$A_{ij} = 0\f$ outside the band | \f$\mathcal{O}(n^2)\f$ |
| `num::assume_tridiagonal(dl, d, du)` | Three occupied diagonals | \f$\mathcal{O}(1)\f$ |
| `num::assume_sparse_csr(A)` | Valid CSR index structure | \f$\mathcal{O}(\text{nnz})\f$ |
| `num::make_square(A)` | \f$\text{rows} = \text{cols}\f$ | \f$\mathcal{O}(1)\f$ |
| `num::make_symmetric(A)` | \f$A = A^T\f$ | \f$\mathcal{O}(n^2)\f$ |
| `num::make_spd(A)` | \f$A = LL^T\f$ exists | \f$\mathcal{O}(n^3)\f$ |

Matrix-free operators use the same taggers under `num::operators`:

```cpp
num::operators::DenseOp Aop(A);

auto spd_op = num::operators::assume_spd(Aop);
num::cg(spd_op, b, x); // Conjugate gradient requires the SPD tag.
```

---

## 3. Compile-Time Enforcement

Passing an untagged matrix to a routine with an invariant precondition is a compile error:

```cpp
num::Matrix A(3, 3, 1.0);
num::Vector b(3, 1.0), x(3, 0.0);

auto f = num::cholesky(A);  // error: A carries no SPD invariant.
auto g = num::lu(A);        // error: A carries no square-dimension invariant.
auto r = num::cg(A, b, x);  // error: A carries no SPD invariant.
```

The diagnostic names the property, the reason it is required, and every way forward:

```text
error: static assertion failed: cg() requires a matrix carrying the SPD invariant:
conjugate gradients minimize a quadratic form that is bounded below only for positive
definite A, and break down silently otherwise. Establish it with num::assume_spd(A) or
num::make_spd(A). For a general matrix use num::gmres(A, ...), which requires no such
invariant. To bypass deliberately, call num::unsafe::cg(A, ...).
```

Routines with no invariant precondition accept raw matrices:

```cpp
num::SolverResult r = num::gmres(A, b, x); // GMRES requires only linearity.
```

### Enforced Routines

| Routine | Required Invariant | Alternative Without It |
| :--- | :--- | :--- |
| `num::cholesky` | SPD | `num::lu` |
| `num::cg` | SPD | `num::gmres` |
| `num::lu` | Square | none |
| `num::eig_sym` | Symmetric | `num::power_iteration` |
| `num::lanczos` | Symmetric | `num::power_iteration` |

---

## 4. num::unsafe

Every enforced routine has an untagged form under `num::unsafe`. It requires no invariant and performs no verification:

```cpp
num::Matrix A(2, 2, 0.0);
A(0, 0) = 1.0;
A(1, 1) = -1.0; // Indefinite.

auto f = num::unsafe::cholesky(A);
// f.success == false. The factorization reports failure instead of throwing.
```

Available: `num::unsafe::cholesky`, `num::unsafe::lu`, `num::unsafe::eig_sym`, `num::unsafe::cg`, `num::unsafe::lanczos`.

The opt-out is a namespace at the call site, so every use is visible in the source and can be found by search.

---

## 5. Execution Presets

Presets control how much work the runtime samplers do. They do not affect compile-time enforcement:

```cpp
num::set_preset(num::preset::strict);     // Basis and randomized probing. Default.
num::set_preset(num::preset::balanced);   // Dimension and structure checks only.
num::set_preset(num::preset::unsafe);     // Probing disabled.
num::set_preset(num::preset::production); // All diagnostics disabled.
```

`num::ScopedPreset` restores the previous preset on scope exit:

```cpp
{
    num::ScopedPreset guard(num::preset::production);

    for (num::idx i = 0; i < iterations; ++i) {
        auto factor = num::cholesky(num::assume_spd(A)); // Tag required. Probing skipped.
    }
} // Previous preset restored.
```

---

## 6. Runtime Verification

A property quantified over all vectors cannot be proved by sampling. The samplers are built to reject violations.

Basis probes test a necessary condition exactly. A positive definite operator satisfies \f$A_{ii} = \langle e_i, A e_i \rangle > 0\f$, and an isometry satisfies \f$\|A e_i\| = 1\f$. Randomized probes then sample away from the coordinate axes using a fixed seed, so a reported violation reproduces exactly.

Definiteness also needs the smallest eigenvalue, because a null space has measure zero and random probing never lands on it:

```cpp
num::Matrix L(2, 2, 0.0); // Graph Laplacian. Symmetric, singular.
L(0, 0) =  1.0; L(0, 1) = -1.0;
L(1, 0) = -1.0; L(1, 1) =  1.0;

auto P = num::assume_psd(L); // Accepted. <x, Lx> >= 0.
auto S = num::assume_spd(L); // Throws. The smallest eigenvalue is 0.
```

Tolerances are relative and derived from the scalar type as \f$\sqrt{\varepsilon}\f$, so the checks hold for `float`, `double`, and `std::complex`.

### Verifying Algebraic Laws

The vector space, inner product, and norm axioms can be sampled directly. The probes route through the type's own operations, so checking `num::Vector` checks the shipped `dot`, `norm`, `axpy`, and `scale` across whichever backend the build selected:

```cpp
num::debug::verify_additive_group_axioms<num::Vector>(64); // Associativity, commutativity,
                                                           // identity, inverses.
num::debug::verify_vector_space_axioms<num::Vector>(64);   // Distributivity, scalar action.
num::debug::verify_inner_product_axioms<num::CVector>(64); // Conjugate symmetry, linearity,
                                                           // positive definiteness.
num::debug::verify_norm_axioms<num::Vector>(64);           // Homogeneity, triangle inequality,
                                                           // ||x||^2 == <x,x>.
num::debug::verify_hilbert_space_axioms<num::Vector>(64);  // All of the above.
```

Conjugate symmetry \f$\langle x,y \rangle = \overline{\langle y,x \rangle}\f$ separates a Hermitian form from an unconjugated bilinear one. The difference is invisible on real data and corrupts every complex Krylov method.

---

## 7. The Property Hierarchy

Operator properties form an inheritance lattice. A type declares one tag and satisfies every weaker concept:

\f[
\text{linear} \subset \text{normal} \subset
\begin{cases}
\text{self\_adjoint} \subset \text{psd} \subset \text{spd} \\
\text{skew\_adjoint} \\
\text{unitary}
\end{cases}
\f]

```cpp
struct MyOperator {
    using properties = num::property::spd; // One declaration.

    num::idx rows() const { return n; }
    num::idx cols() const { return n; }
    void apply(const num::Vector &x, num::Vector &y) const;
};

static_assert(num::SPDOperator<MyOperator>);
static_assert(num::PSDOperator<MyOperator>);          // Implied.
static_assert(num::SelfAdjointOperator<MyOperator>);  // Implied.
static_assert(num::NormalOperator<MyOperator>);       // Implied.
static_assert(!num::SkewAdjointOperator<MyOperator>);
```

Each concept is a conjunction containing the next weaker one, so a solver accepts any tag at least as strong as the one it requires:

```cpp
auto sym = num::operators::assume_symmetric(Aop);
auto spd = num::operators::assume_spd(Aop);

num::gmres(sym, b, x);  // Linear is enough.
num::minres(sym, b, x); // Self-adjoint is required and present.
num::cg(spd, b, x);     // SPD is required and present.
num::cg(sym, b, x);     // error: self-adjoint does not imply positive definite.
```

Stored matrices and matrix-free operators use the same lattice, so `num::SPDMatrixLike<M>` and `num::SPDOperator<Op>` state the same property about different representations.

---

## 8. Algebraic Structure

Section 1 introduced `num::VectorSpace`. The full set of structures, from weakest to strongest:

| Concept | Adds |
| :--- | :--- |
| `num::Field<T>` | A scalar type with `+`, `-`, `*`, `/` over a floating-point field |
| `num::AdditiveGroup<V>` | Closure under addition, a zero, and inverses |
| `num::VectorSpace<V>` | Scalar multiplication compatible with the field |
| `num::InnerProductSpace<V>` | An inner product \f$\langle x,y \rangle\f$ |
| `num::NormedSpace<V>` | A norm \f$\|x\|\f$ |
| `num::HilbertSpace<V>` | Both, with \f$\|x\|^2 = \langle x,x \rangle\f$ |

Closure is the requirement that indexing alone misses. A type must be able to construct an element and receive a sum:

```cpp
static_assert(num::VectorSpace<num::Vector>);
static_assert(num::VectorSpace<num::CVector>);            // Complex field.
static_assert(num::VectorSpace<std::vector<float>>);      // Foreign container.
static_assert(!num::VectorSpace<std::span<const double>>);// A view has nowhere to put x+y.
static_assert(!num::VectorSpace<std::vector<int>>);       // Integers form a ring.
```

Storage layout is described separately under `num::repr`:

```cpp
static_assert(num::repr::Contiguous<num::Vector>);
static_assert(num::repr::DenseRowMajor<num::Matrix>);
static_assert(num::repr::CSR<num::SparseMatrix>);
static_assert(num::repr::Banded<num::BandedMatrix>);
```

A kernel that needs a pointer and a stride constrains on `num::repr`. An algorithm that needs a vector space constrains on `num::VectorSpace`.

---

## 9. Solver Coverage

| Concept | Invariant | Required By | Verifier |
| :--- | :--- | :--- | :--- |
| `SPDMatrixLike<M>` | \f$\mathbf{x}^T A\mathbf{x} > 0\f$ | `cholesky`, `cg` | `verify_spd_sample` |
| `PSDMatrixLike<M>` | \f$\mathbf{x}^T A\mathbf{x} \geq 0\f$ | Laplacians, Gram matrices | `verify_psd_sample` |
| `SymmetricMatrixLike<M>` | \f$A = A^T\f$ | `eig_sym`, `lanczos` | `verify_symmetry_sample` |
| `SquareMatrixLike<M>` | \f$A: V \to V\f$ | `lu`, `det`, `inv`, `expv` | `verify_square` |
| `BandedMatrixLike<B>` | \f$A_{ij}=0,\ j-i \notin [-k_l, k_u]\f$ | `banded_solve` | `verify_band_occupancy` |
| `TridiagonalMatrixLike<T>` | Three occupied diagonals | `thomas` | `verify_tridiagonal_structure` |
| `SparseMatrixCSRLike<M>` | Monotonic offsets, in-range columns | `spmv`, `approxchol` | `verify_sparse_structure` |
| `SPDOperator<Op>` | \f$\langle x, Ax\rangle > 0\f$ | `cg`, `pcg` | `verify_spd_sample` |
| `SelfAdjointOperator<Op>` | \f$A = A^*\f$ | `minres`, `lanczos` | `verify_symmetry_sample` |
| `NormalOperator<Op>` | \f$AA^* = A^*A\f$ | Spectral methods | inherited |
| `UnitaryOperator<Op>` | \f$A^*A = I\f$ | `fft`, Givens rotations | `verify_orthogonal_sample` |
| `ProjectionOperator<Op>` | \f$P^2 = P = P^*\f$ | Subspace projection | `verify_projection_sample` |
| `SkewAdjointOperator<Op>` | \f$A = -A^*\f$ | Advection, Poisson brackets | `verify_skew_symmetry_sample` |
| `AdjointableLinearOperator<Op>` | \f$\langle Ax,y\rangle = \langle x,A^*y\rangle\f$ | `lsqr`, `gmres` | `verify_adjoint_sample` |
| `LinearOperator<Op>` | \f$A(\alpha x + \beta y) = \alpha Ax + \beta Ay\f$ | `gmres`, `expv`, `power_iteration` | `verify_linearity_sample` |
| `TriangularFactor<F>` | Admits `solve_in_place` | Substitution solves | none |
| `DirectFactorization<F>` | Reusable across right-hand sides | `lu_solve`, `cholesky_solve` | none |
| `Preconditioner<M>` | \f$z \approx M^{-1}r\f$ | `pcg`, preconditioned `gmres` | none |
| `IsODEProblem<P>` | \f$\dot y = f(t,y)\f$ | `Euler`, `RK4`, `RK45` | `verify_order_of_accuracy` |
| `IsSymplecticODEProblem<P>` | Separable \f$H(q,p)\f$ | Verlet, Forest-Ruth | `verify_symplectic_2form` |
| `IncidenceStructure<G>` | \f$\sum_u \deg u = 2\lvert E\rvert\f$ | BFS, Dijkstra, Kruskal | `verify_handshake_lemma` |
| `linear::LaplacianGraph<G>` | \f$\Delta\mathbf{1}=0\f$, PSD | `approxchol`, spectral clustering | `verify_laplacian_structure` |
| `EquivalenceRelation<T>` | Reflexive, symmetric, transitive | Kruskal, connected components | `verify_equivalence_relation` |
| `AddressablePriorityQueue<T>` | `top_key()` is minimal | Dijkstra, degree queues | `verify_heap_order` |

For where these concepts live and how the modules layer, see @ref page_architecture "Library Layout".

---

## 10. Worked Example

@example 14_concepts_and_property_invariants.cpp

```cpp
#include <numerics.hpp>

int main() {
    using namespace num;

    Matrix A(4, 4, 0.0);
    for (idx i = 0; i < 4; ++i) {
        A(i, i) = 4.0;
        if (i > 0) {
            A(i, i - 1) = 1.0;
            A(i - 1, i) = 1.0;
        }
    }
    Vector b{1.0, 2.0, 3.0, 4.0};

    // auto bad = cholesky(A); // error: A carries no SPD invariant.

    auto spd_A = assume_spd(A);
    static_assert(SPDMatrixLike<decltype(spd_A)>);
    static_assert(SymmetricMatrixLike<decltype(spd_A)>); // Implied by the lattice.
    static_assert(SquareMatrixLike<decltype(spd_A)>);    // Implied by the lattice.

    auto chol = cholesky(spd_A);
    Vector x(4, 0.0);
    cholesky_solve(chol, b, x);

    operators::DenseOp Aop(A);
    auto spd_op = operators::assume_spd(Aop);
    static_assert(SPDOperator<decltype(spd_op)>);

    Vector x_cg(4, 0.0);
    cg(spd_op, b, x_cg); // Selects CG over MINRES and GMRES.

    Matrix indefinite(2, 2, 0.0);
    indefinite(0, 0) = 1.0;
    indefinite(1, 1) = -1.0;
    auto f = unsafe::cholesky(indefinite); // f.success == false.
}
```
