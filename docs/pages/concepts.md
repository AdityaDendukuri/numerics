# Concepts, Invariants & Diagnostics {#page_concepts}

Concepts in this library express two kinds of requirement.

A **structural** requirement is decided by the compiler. Whether a type has an `apply`
member taking a vector of its scalar field is a syntactic question, and a `requires`
clause answers it.

A **law** is asserted by the caller. Self-adjointness and positive definiteness cannot be
determined from a type. The caller states the claim, the library records it in the
`num::law` hierarchy, and a runtime probe samples it. Sampling rejects false claims. It
does not prove true ones.

Every concept below is a structural requirement, a law, or a conjunction of the two. Every
concept is also a refinement of another concept. The test `ConceptHierarchy` parses these
headers and fails the build if a concept is asserted against raw syntax instead of
refining another concept, or if it names a concrete container instead of taking the space
as a template parameter.

---

## 1. The hierarchy

The concepts refine one another, starting from a scalar field. Each is the one above it
equipped with a further operation, or carrying a further law: a normed space is a vector
space equipped with a norm, an inner product space is a normed space equipped with an
inner product, and a self-adjoint operator is an endomorphism satisfying \f$A = A^*\f$.

```
field<T>                    + - * /, 0, 1
 └ additive_group<V>         dimension, zero_like, copy
    └ vector_space<V>        + scale, axpy                 ── law::vector_space
       └ normed_space<V>     + norm                        ── law::normed_space
          └ inner_product_space<V>  + inner                 ── law::inner_product_space
             └ hilbert_space<V>    ‖x‖² = ⟨x,x⟩            ── law::hilbert_space

linear_map<Op>               domain and codomain are spaces ── law::linear_map
 └ linear_operator<Op>       + apply, rows, cols
    └ endomorphism<Op>      domain == codomain             ── law::endomorphism
       └ normal_operator     AA* = A*A                      ── law::normal
          ├ self_adjoint_operator  A = A*                    ── law::self_adjoint
          │  └ psd_operator        ⟨x,Ax⟩ ≥ 0               ── law::psd
          │     ├ spd_operator      ⟨x,Ax⟩ > 0              ── law::spd
          │     └ projection_operator  P = P* = P²          ── law::projection
          ├ skew_adjoint_operator  A = −A*                   ── law::skew_adjoint
          └ unitary_operator      A*A = I                   ── law::unitary
```

The hierarchy is defined in `core/math/concepts.hpp` and re-exported into `num`.
`num::vector_space` and `num::math::vector_space` name the same concept.

### Space concepts

| Concept | Requires |
| :--- | :--- |
| `num::scalar<T>` | `+`, `-`, `*`, `/` over a floating-point base. Satisfied by `double`, `float`, and `std::complex` of either. A synonym for `num::field`. |
| `num::additive_group<V>` | `scalar<scalar_t<V>>`, copy construction, `dimension(v)`, `zero_like(v)` |
| `num::vector_space<V>` | `additive_group<V>`, `scale(a, v)`, `axpy(a, x, v)` |
| `num::normed_space<V>` | `vector_space<V>`, `norm(v)` |
| `num::inner_product_space<V>` | `normed_space<V>`, `inner(x, y)` |
| `num::hilbert_space<V>` | `inner_product_space<V>` whose norm satisfies \f$\Vert x \Vert^2 = \langle x, x \rangle\f$ |

```cpp
static_assert(num::vector_space<num::vec>);
static_assert(num::vector_space<num::cvec>);
static_assert(num::vector_space<std::vector<float>>);       // Foreign type
static_assert(!num::vector_space<std::vector<int>>);        // Integers form a ring, not a field
```

Storage layout is described separately, under `num::repr`. Bandedness is a property of
memory, not of a linear map.

```cpp
static_assert(num::repr::contiguous<num::vec>);
static_assert(num::repr::dense_row_major<num::mat>);
static_assert(num::repr::csr<num::spmat>);
static_assert(num::repr::banded<num::band_mat>);
```

---

## 2. Operator Concepts

The laws are partially ordered by implication, and inheritance encodes that order. A type
claiming `law::spd` therefore satisfies `self_adjoint_operator` and every weaker concept
with no further declaration. The order is not a lattice: `law::spd` and `law::unitary`
have a greatest lower bound (`law::normal`) but no least upper bound, since no law implies
both.

`law::projection` derives from `law::psd`. A projector is positive semidefinite because
\f$\langle x, Px \rangle = \langle x, P^2 x \rangle = \Vert Px \Vert^2 \ge 0\f$.

\f[
\text{linear} \subset \text{endomorphism} \subset \text{normal} \subset
\begin{cases}
\text{self-adjoint} \subset \text{psd} \subset
  \begin{cases}\text{spd} \\ \text{projection}\end{cases} \\
\text{skew-adjoint} \\
\text{unitary}
\end{cases}
\f]

| Concept | Definition | Refines |
| :--- | :--- | :--- |
| `num::linear_map<Op>` | \f$A: X \to Y\f$ between vector spaces | — |
| `num::linear_operator<Op>` | `linear_map` with `apply`, `rows`, `cols` | `num::linear_map` |
| `num::endomorphism<Op>` | domain equals codomain | `num::linear_operator` |
| `num::normal_operator<Op>` | \f$A A^* = A^* A\f$ | `num::endomorphism` |
| `num::self_adjoint_operator<Op>` | \f$A = A^*\f$ | `num::normal_operator` |
| `num::psd_operator<Op>` | \f$\langle x, A x \rangle \ge 0\f$ | `num::self_adjoint_operator` |
| `num::spd_operator<Op>` | \f$\langle x, A x \rangle > 0\f$ for \f$x \ne 0\f$ | `num::psd_operator` |
| `num::projection_operator<Op>` | \f$P = P^* = P^2\f$ | `num::psd_operator` |
| `num::skew_adjoint_operator<Op>` | \f$A = -A^*\f$ | `num::normal_operator` |
| `num::unitary_operator<Op>` | \f$A^* A = I\f$ | `num::normal_operator` |

Each concept takes its domain and codomain from the operator's associated types.
`num::spd_operator<Op>` and `num::spd_operator<Op, num::vec, num::vec>` are the same
concept with the spaces left implicit or stated.

---

## 3. Declaring a type's laws

A type declares its laws with a member typedef. The hierarchy supplies the implications, so
`law::spd` alone also provides `self_adjoint`, `normal`, `endomorphism`, and `linear_map`.

An operator also declares the spaces it maps between. The hierarchy reads them to check that
the domain and codomain are vector spaces.

```cpp
struct Custom1DLaplacian {
    using math_laws = num::math::type_list<num::law::spd>;
    using domain_type = num::vec;
    using codomain_type = num::vec;

    num::idx n;
    [[nodiscard]] num::idx rows() const noexcept { return n; }
    [[nodiscard]] num::idx cols() const noexcept { return n; }

    void apply(const num::vec &x, num::vec &y) const {
        for (num::idx i = 0; i < n; ++i) {
            y[i] = 2.0 * x[i] - (i > 0 ? x[i - 1] : 0.0) - (i + 1 < n ? x[i + 1] : 0.0);
        }
    }
};

static_assert(num::spd_operator<Custom1DLaplacian>);
```

For a type you do not control, such as a standard container or a third-party matrix,
specialize `claims_of` instead.

```cpp
namespace num::math {
template <> struct claims_of<ThirdPartyMatrix> {
    using type = type_list<law::self_adjoint>;
};
}
```

A type never acquires a law from its syntax alone. `std::string` defines `operator+` and
is not a vector space.

---

## 4. Diagnostics

A missing or insufficient law is reported at the call site. The output below is taken from
real compiler and program runs, trimmed to the lines that identify the cause.

### No law claimed

```cpp
// DOES NOT COMPILE -- the diagnostic is below.
mat A(3, 3, 1.0);
vec b(3, 1.0), x(3, 0.0);
cg(operators::dense_op(A), b, x, 1e-10, 100);
```

```
error: no matching function for call to 'cg'
cg.hpp:151: note: because 'claims<num::operators::dense_op, law::spd>' evaluated to false
```

`dense_op` claims only `law::linear_map`. Attach the stronger claim with
`num::operators::assume_spd(...)`.

### Law too weak

```cpp
// DOES NOT COMPILE -- the diagnostic is below.
auto sym = operators::assume_symmetric(operators::dense_op(A));
cg(sym, b, x, 1e-10, 100);
```

```
error: no matching function for call to 'cg'
cg.hpp:151: note: because 'claims<structured_op<dense_op, law::self_adjoint>, law::spd>'
                  evaluated to false
```

The wrapper carries its law in its type, so the message shows the law claimed next to the
law required. `num::minres` accepts this operator. `num::cg` does not.

### Type outside the hierarchy

```cpp
// DOES NOT COMPILE -- the diagnostic is below.
static_assert(num::vector_space<std::vector<int>>);
```

```
error: static assertion failed
note: because 'std::vector<int>' does not satisfy 'vector_space'
concepts.hpp:82: note: because 'std::vector<int>' does not satisfy 'additive_group'
concepts.hpp:74: note: because 'scalar_t<vector<int>>' (aka 'int') does not satisfy 'field'
```

The notes descend the hierarchy to the cause. The integers form a ring rather than a field, so
`std::vector<int>` satisfies no level. `std::vector<double>` and `std::vector<float>` do.

### Spaces not declared

```cpp
// DOES NOT COMPILE -- the diagnostic is below.
struct MyOp {
    using math_laws = num::math::type_list<num::law::spd>;
    // no domain_type / codomain_type
    ...
};
static_assert(num::spd_operator<MyOp>);
```

```
note: because 'MyOp' does not satisfy 'spd_operator'
concepts.hpp:184: note: because 'psd_operator<MyOp, void, void>' evaluated to false
concepts.hpp:180: note: because 'self_adjoint_operator<MyOp, void, void>' evaluated to false
```

`void, void` indicates that the domain and codomain defaulted to nothing because the type
declares neither. Add the two typedefs, or name the spaces at the call site as
`num::spd_operator<MyOp, num::vec, num::vec>`.

### False claim, rejected at run time

The compiler checks that a law was claimed. It cannot check that the claim is true. Two
runtime layers do.

Attaching a claim samples it, along with every weaker law.

```cpp
// Compiles; throws at run time. The output is below.
mat A(3, 3, 0.0);
A(0,0)=2; A(1,1)=2; A(2,2)=2; A(0,1)=5.0; A(1,0)=-5.0;   // not symmetric
auto spd = operators::assume_spd(operators::dense_op(A));
```

```
[PropertyError] Error at example.cpp:7 in int main():
  assume_symmetric() assertion failed: relative |<x,Ay> - conj(<y,Ax>)| = 1.649990
  on probe 0 exceeds tolerance 0.000000, so the operator is NOT self-adjoint.
```

The self-adjointness probe reports the failure, not the definiteness probe. `assume_spd`
verifies the whole chain beneath `law::spd`, weakest law first. The message names the
source line that made the claim.

Sampling can miss a violation. The algorithm checks its own invariant as it runs. An
operator that claims `law::spd` and applies an indefinite map compiles, passes the probes,
and then fails inside the solver:

```
caught: cg: positive-definite curvature invariant was violated
```

This check costs \f$\mathcal{O}(1)\f$ per iteration against \f$\mathcal{O}(n)\f$ of work,
and `NDEBUG` does not remove it.

---

## 5. Attaching evidence to a value

A law describes a type. Evidence describes a particular matrix or operator. The caller
attaches it explicitly.

```cpp
auto claimed   = num::assume_spd(A);              // Claim; sampled under the active preset.
auto validated = num::make_spd(A);                // Validated by Cholesky, O(n^3); throws on failure.
auto by_law    = num::assume<num::law::spd>(A);   // Claim, stated as a law.
auto checked   = num::require<num::law::spd>(A);  // Validated exhaustively where a validator exists.
```

`assume` records `evidence_origin::assumed`. `require` records `evidence_origin::verified`.
Both store the source location of the call, which the diagnostics above report.

| mat tagger | Operator tagger | Invariant | Cost |
| :--- | :--- | :--- | :--- |
| `assume_square(A)` | — | \f$\text{rows} = \text{cols}\f$ | \f$\mathcal{O}(1)\f$ |
| `assume_symmetric(A)` | `operators::assume_symmetric(op)` | \f$A = A^T\f$ | Sampled |
| `assume_psd(A)` | `operators::assume_psd(op)` | \f$x^T A x \ge 0\f$ | Sampled |
| `assume_spd(A)` | `operators::assume_spd(op)` | \f$x^T A x > 0\f$ | Sampled |
| `assume_banded(A, kl, ku)` | — | Occupancy inside the band | \f$\mathcal{O}(n^2)\f$ |
| `assume_tridiagonal(...)` | — | Three occupied diagonals | \f$\mathcal{O}(1)\f$ |
| `assume_sparse_csr(A)` | — | Monotonic offsets, valid indices | \f$\mathcal{O}(\text{nnz})\f$ |

---

## 6. Bypassing enforcement

Each enforced solver has a counterpart under `num::unsafe` that takes an untagged argument
and performs no invariant check. These report failure through their return value rather
than by throwing.

```cpp
num::mat indefinite(2, 2, 0.0);
indefinite(0, 0) = 1.0;
indefinite(1, 1) = -1.0;

auto factor = num::unsafe::cholesky(indefinite); // factor.success == false
```

Available: `num::unsafe::cholesky`, `num::unsafe::lu`, `num::unsafe::eig_sym`,
`num::unsafe::cg`, `num::unsafe::lanczos`.

---

## 7. Diagnostic presets

Two separate dials control checking. `NUMERICS_DIAGNOSTICS` decides at compile time what
checking code exists. The runtime preset decides whether it runs.

### The compile-time ceiling

| `NUMERICS_DIAGNOSTICS` | Contains | Default in |
| :--- | :--- | :--- |
| `0` | nothing; every check and probe is discarded | — |
| `1` | shape checks: dimensions, emptiness, finiteness | builds with `NDEBUG` |
| `2` | property sampling as well: symmetry, definiteness, linearity | builds without `NDEBUG` |

Property sampling costs \f$\mathcal{O}(n^2)\f$ in the operator's size, so it is not a
Release default. Attaching an SPD claim to a 1024×1024 operator samples at roughly 31 ms
against a 3.4 ms conjugate-gradient solve of the same system. Under the default ceiling
the same call costs nothing measurable:

```
ceiling=1   assume_spd =  0.00 ms   cg = 3.38 ms
ceiling=2   assume_spd = 31.19 ms   cg = 3.39 ms
```

At ceiling 1 the probes are not merely skipped, they are absent. No `verify_*_sample`
symbol is emitted, and the object file is roughly a quarter the size.

Build with `-DNUMERICS_DIAGNOSTICS=2` to keep sampling in an optimized build, which is
what a test suite or a numerically suspicious run wants. This library's own test suite
sets it.

### The runtime preset

```cpp
num::set_preset(num::preset::strict);     // Sample every property.
num::set_preset(num::preset::balanced);   // Shape and dimension checks only.
num::set_preset(num::preset::unsafe);     // Everything off; keep warnings quiet.
num::set_preset(num::preset::production); // Everything off.
```

A request is clamped to the ceiling. Asking for `strict` in a build compiled at level 1
leaves sampling off, because that code was never emitted. `num::get_preset()` still
reports what was asked for, and `num::preset_fully_applied()` reports whether the request
was met:

```cpp
num::set_preset(num::preset::strict);
if (!num::preset_fully_applied()) {
    // Built below the requested level. Rebuild with -DNUMERICS_DIAGNOSTICS=2.
}
```

Lowering is always honoured, so any build can turn diagnostics off.

`num::scoped_preset` restores the previous preset when it goes out of scope.

```cpp
{
    num::scoped_preset guard(num::preset::production);
    for (num::idx i = 0; i < 1000; ++i) {
        auto factor = num::cholesky(num::assume_spd(A)); // Tag attached, probing skipped.
    }
}
```

---

## 8. Axiom verification suites

These functions test a type against the axioms of a space over randomly generated
elements of the given dimension. They are intended for a type you are adding to the hierarchy.

```cpp
num::debug::verify_additive_group_axioms<num::vec>(64);
num::debug::verify_vector_space_axioms<num::vec>(64);
num::debug::verify_inner_product_axioms<num::cvec>(64);
num::debug::verify_norm_axioms<num::vec>(64);
num::debug::verify_hilbert_space_axioms<num::vec>(64);
```

---

## 9. Requirements by routine

| Routine | Requires | Alternative for weaker input |
| :--- | :--- | :--- |
| `num::cholesky` | `num::spd_matrix_like` | `num::lu` |
| `num::cg` | `num::spd_operator` | `num::minres`, `num::gmres` |
| `num::pcg` | `num::spd_operator` for operator and preconditioner | `num::gmres` |
| `num::minres` | `num::self_adjoint_operator` | `num::gmres` |
| `num::gmres` | `num::linear_operator` | — |
| `num::lu` | `num::square_matrix_like` | `num::qr` |
| `num::eig_sym` | `num::symmetric_matrix_like` | `num::power_iteration` |
| `num::lanczos` | `num::self_adjoint_operator` | `num::power_iteration` |

---

## Example

@example 14_concepts_and_property_invariants.cpp

---

## See also

* @ref page_concept_index "Concept Index" — all 83 concepts, grouped by what they describe
* @ref page_kernel_index "num::kernel Index" — the computational half of the library
