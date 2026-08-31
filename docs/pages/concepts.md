# Concepts, Invariants & Diagnostics {#page_concepts}

Compile-time concepts, runtime evidence tags, diagnostic presets, and axiom verification.

---

## 1. Algebraic Concepts (Type Structure)

Algebraic concepts check operations and type-level laws at compile time.

| Concept | Required Operations & Laws |
| :--- | :--- |
| `num::Scalar<T>` | Scalar arithmetic `+`, `-`, `*`, `/` over floating-point base (`double`, `float`, `std::complex`) |
| `num::AdditiveGroup<V>` | Vector addition, negation, and additive zero |
| `num::VectorSpace<V>` | `AdditiveGroup<V>` + scalar multiplication compatible with field |
| `num::InnerProductSpace<V>` | `VectorSpace<V>` + inner product `inner(x, y)` (\f$\langle x, y \rangle = \overline{\langle y, x \rangle}\f$) |
| `num::NormedSpace<V>` | `VectorSpace<V>` + norm `norm_of(x)` (\f$\Vert x \Vert \ge 0\f$, \f$\Vert a x \Vert = \lvert a \rvert \Vert x \Vert\f$) |
| `num::HilbertSpace<V>` | `InnerProductSpace<V>` + `NormedSpace<V>` with \f$\Vert x \Vert^2 = \langle x, x \rangle\f$ |

```cpp
static_assert(num::VectorSpace<num::Vector>);
static_assert(num::VectorSpace<num::CVector>);
static_assert(num::VectorSpace<std::vector<float>>);       // Foreign type
static_assert(!num::VectorSpace<std::span<const double>>); // Non-owning views cannot hold sums
static_assert(!num::VectorSpace<std::vector<int>>);        // Integers form a ring, not a field
```

Storage layout concepts are declared under `num::repr`:
```cpp
static_assert(num::repr::Contiguous<num::Vector>);
static_assert(num::repr::DenseRowMajor<num::Matrix>);
static_assert(num::repr::CSR<num::SparseMatrix>);
static_assert(num::repr::Banded<num::BandedMatrix>);
```

---

## 2. Operator Hierarchy

Properties of linear maps form an inheritance lattice. Declaring a stronger property tag automatically satisfies every weaker concept in the hierarchy:

\f[
\text{linear} \subset \text{normal} \subset
\begin{cases}
\text{self-adjoint} \subset \text{psd} \subset \text{spd} \\
\text{skew-adjoint} \\
\text{unitary}
\end{cases}
\f]

| Concept | Mathematical Definition | Implied Concepts |
| :--- | :--- | :--- |
| `num::LinearOperator<Op>` | \f$A(\alpha x + \beta y) = \alpha A x + \beta A y\f$ | — |
| `num::NormalOperator<Op>` | \f$A A^* = A^* A\f$ | `num::LinearOperator` |
| `num::SelfAdjointOperator<Op>` | \f$A = A^*\f$ | `num::NormalOperator` |
| `num::PSDOperator<Op>` | \f$\langle x, A x \rangle \ge 0\f$ | `num::SelfAdjointOperator` |
| `num::SPDOperator<Op>` | \f$\langle x, A x \rangle > 0 \quad (x \ne 0)\f$ | `num::PSDOperator` |
| `num::SkewAdjointOperator<Op>` | \f$A = -A^*\f$ | `num::NormalOperator` |
| `num::UnitaryOperator<Op>` | \f$A^* A = I\f$ | `num::NormalOperator` |

---

## 3. Value Evidence & Invariant Tags

Properties of a particular matrix or operator cannot be decided by inspecting its type. The caller attaches evidence explicitly:

```cpp
// 1. Tag by claim (sampled under active preset)
auto spd_A = num::assume_spd(A);

// 2. Tag by exhaustive validation (computes Cholesky factorization in O(n^3); throws on failure)
auto spd_A = num::make_spd(A);

// 3. New evidence vocabulary (core/math)
auto spd_A = num::assume<num::axiom::positive_definite>(A);
auto spd_A = num::require<num::axiom::positive_definite>(A);
```

### Available Taggers

| Matrix Tagger | Operator Tagger | Invariant | Sampling Cost |
| :--- | :--- | :--- | :--- |
| `assume_square(A)` | — | \f$\text{rows} = \text{cols}\f$ | \f$\mathcal{O}(1)\f$ |
| `assume_symmetric(A)` | `operators::assume_symmetric(op)` | \f$A = A^T\f$ | Sampled |
| `assume_psd(A)` | `operators::assume_psd(op)` | \f$x^T A x \ge 0\f$ | Sampled |
| `assume_spd(A)` | `operators::assume_spd(op)` | \f$x^T A x > 0\f$ | Sampled |
| `assume_banded(A, kl, ku)` | — | Occupancy inside band | \f$\mathcal{O}(n^2)\f$ |
| `assume_tridiagonal(...)` | — | Three occupied diagonals | \f$\mathcal{O}(1)\f$ |
| `assume_sparse_csr(A)` | — | Monotonic offsets, valid indices | \f$\mathcal{O}(\text{nnz})\f$ |

---

## 4. num::unsafe Bypass

Every enforced solver has an untagged counterpart under `num::unsafe` that skips invariant checks:

```cpp
num::Matrix indefinite(2, 2, 0.0);
indefinite(0, 0) = 1.0;
indefinite(1, 1) = -1.0;

auto factor = num::unsafe::cholesky(indefinite); // factor.success == false (no exception)
```

Available: `num::unsafe::cholesky`, `num::unsafe::lu`, `num::unsafe::eig_sym`, `num::unsafe::cg`, `num::unsafe::lanczos`.

---

## 5. Diagnostic Presets

Presets control runtime sampling depth without affecting compile-time enforcement:

```cpp
num::set_preset(num::preset::strict);     // Randomized basis probing (default)
num::set_preset(num::preset::balanced);   // Shape and dimension checks only
num::set_preset(num::preset::unsafe);     // Probing disabled
num::set_preset(num::preset::production); // All diagnostics disabled
```

Scoped preset guard:
```cpp
{
    num::ScopedPreset guard(num::preset::production);
    for (num::idx i = 0; i < 1000; ++i) {
        auto factor = num::cholesky(num::assume_spd(A)); // Tag attached; probing skipped
    }
} // Previous preset restored on scope exit
```

---

## 6. Axiom Verification Suites

```cpp
num::debug::verify_additive_group_axioms<num::Vector>(64);
num::debug::verify_vector_space_axioms<num::Vector>(64);
num::debug::verify_inner_product_axioms<num::CVector>(64);
num::debug::verify_norm_axioms<num::Vector>(64);
num::debug::verify_hilbert_space_axioms<num::Vector>(64);
```

---

## 7. Concept and Solver Mapping

| Routine | Required Concept / Invariant | Fallback / General Alternative |
| :--- | :--- | :--- |
| `num::cholesky` | `num::SPDMatrixLike` | `num::lu` |
| `num::cg` | `num::SPDOperator` | `num::gmres`, `num::minres` |
| `num::pcg` | `num::SPDOperator` (operator and preconditioner) | `num::gmres` |
| `num::minres` | `num::SelfAdjointOperator` | `num::gmres` |
| `num::gmres` | `num::LinearOperator` | — |
| `num::lu` | `num::SquareMatrixLike` | `num::qr` |
| `num::eig_sym` | `num::SymmetricMatrixLike` | `num::power_iteration` |
| `num::lanczos` | `num::SelfAdjointOperator` | `num::power_iteration` |

---

## Example

@example 14_concepts_and_property_invariants.cpp

