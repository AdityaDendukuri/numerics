# Concepts, Invariants & Diagnostics {#page_concepts}

Concepts express two kinds of requirement. **Structure** the compiler decides. **Laws** the
caller asserts and a probe samples.

```cpp
// Structure is syntax, so a requires-clause settles it outright.
static_assert( num::vector_space<num::vec>);
static_assert(!num::vector_space<std::vector<int>>);   // integers are a ring, not a field

// A law cannot be read off a type. The caller states it.
num::mat A = num::identity(4);
auto op  = num::operators::dense_op(A);                // claims law::linear_map, nothing more
auto sym = num::operators::assume_symmetric(op);       // now claims law::self_adjoint
static_assert(!num::self_adjoint_operator<decltype(op)>);
static_assert( num::self_adjoint_operator<decltype(sym)>);
```

---

## All concepts

Grouped by what each describes. Every one is a structural requirement the compiler
decides, a law the caller asserts, or a conjunction of the two, and every one refines
another.

<div class="sym-index">
<div class="kidx-group"><span class="kidx-title">Scalars</span><br/><span class="kidx-syms"><a href="conceptnum_1_1scalar.html">scalar</a> &ndash; <a href="conceptnum_1_1math_1_1field.html">field</a> &ndash; <a href="conceptnum_1_1differentiable__function.html">differentiable_function</a> &ndash; <a href="conceptnum_1_1scalar__function.html">scalar_function</a></span></div><div class="kidx-group"><span class="kidx-title">Spaces</span><br/><span class="kidx-syms"><a href="conceptnum_1_1math_1_1additive__group.html">additive_group</a> &ndash; <a href="conceptnum_1_1math_1_1vector__space.html">vector_space</a> &ndash; <a href="conceptnum_1_1math_1_1normed__space.html">normed_space</a> &ndash; <a href="conceptnum_1_1math_1_1inner__product__space.html">inner_product_space</a> &ndash; <a href="conceptnum_1_1math_1_1hilbert__space.html">hilbert_space</a> &ndash; <a href="conceptnum_1_1math_1_1mutable__vector__space.html">mutable_vector_space</a> &ndash; <a href="conceptnum_1_1math_1_1contiguous__vector.html">contiguous_vector</a> &ndash; <a href="conceptnum_1_1math_1_1linear__subspace__of.html">linear_subspace_of</a></span></div><div class="kidx-group"><span class="kidx-title">Maps and operators</span><br/><span class="kidx-syms"><a href="conceptnum_1_1math_1_1linear__map.html">linear_map</a> &ndash; <a href="conceptnum_1_1math_1_1linear__operator.html">linear_operator</a> &ndash; <a href="conceptnum_1_1math_1_1adjointable__linear__operator.html">adjointable_linear_operator</a> &ndash; <a href="conceptnum_1_1math_1_1endomorphism.html">endomorphism</a> &ndash; <a href="conceptnum_1_1math_1_1endomorphism__on.html">endomorphism_on</a> &ndash; <a href="conceptnum_1_1math_1_1nonlinear__operator.html">nonlinear_operator</a> &ndash; <a href="conceptnum_1_1sparse__convertible.html">sparse_convertible</a></span></div><div class="kidx-group"><span class="kidx-title">Operator laws</span><br/><span class="kidx-syms"><a href="conceptnum_1_1math_1_1normal__operator.html">normal_operator</a> &ndash; <a href="conceptnum_1_1math_1_1self__adjoint__operator.html">self_adjoint_operator</a> &ndash; <a href="conceptnum_1_1math_1_1psd__operator.html">psd_operator</a> &ndash; <a href="conceptnum_1_1math_1_1spd__operator.html">spd_operator</a> &ndash; <a href="conceptnum_1_1math_1_1projection__operator.html">projection_operator</a> &ndash; <a href="conceptnum_1_1math_1_1skew__adjoint__operator.html">skew_adjoint_operator</a> &ndash; <a href="conceptnum_1_1math_1_1unitary__operator.html">unitary_operator</a></span></div><div class="kidx-group"><span class="kidx-title">Matrices</span><br/><span class="kidx-syms"><a href="conceptnum_1_1matrix__space.html">matrix_space</a> &ndash; <a href="conceptnum_1_1mutable__matrix__space.html">mutable_matrix_space</a> &ndash; <a href="conceptnum_1_1square__matrix__like.html">square_matrix_like</a> &ndash; <a href="conceptnum_1_1symmetric__matrix__like.html">symmetric_matrix_like</a> &ndash; <a href="conceptnum_1_1psd__matrix__like.html">psd_matrix_like</a> &ndash; <a href="conceptnum_1_1spd__matrix__like.html">spd_matrix_like</a> &ndash; <a href="conceptnum_1_1banded__matrix__like.html">banded_matrix_like</a> &ndash; <a href="conceptnum_1_1tridiagonal__matrix__like.html">tridiagonal_matrix_like</a> &ndash; <a href="conceptnum_1_1sparse__matrix__csr__like.html">sparse_matrix_csr_like</a> &ndash; <a href="conceptnum_1_1triangular__factor.html">triangular_factor</a></span></div><div class="kidx-group"><span class="kidx-title">Storage layout</span><br/><span class="kidx-syms"><a href="conceptnum_1_1repr_1_1contiguous.html">contiguous</a> &ndash; <a href="conceptnum_1_1repr_1_1dense__row__major.html">dense_row_major</a> &ndash; <a href="conceptnum_1_1repr_1_1csr.html">csr</a> &ndash; <a href="conceptnum_1_1repr_1_1banded.html">banded</a> &ndash; <a href="conceptnum_1_1repr_1_1tridiagonal.html">tridiagonal</a></span></div><div class="kidx-group"><span class="kidx-title">Claims and evidence</span><br/><span class="kidx-syms"><a href="conceptnum_1_1claims.html">claims</a> &ndash; <a href="conceptnum_1_1math_1_1law__tag.html">law_tag</a> &ndash; <a href="conceptnum_1_1math_1_1mathematical__proposition.html">mathematical_proposition</a></span></div><div class="kidx-group"><span class="kidx-title">Solvers and preconditioners</span><br/><span class="kidx-syms"><a href="conceptnum_1_1direct__factorization.html">direct_factorization</a> &ndash; <a href="conceptnum_1_1preconditioner.html">preconditioner</a> &ndash; <a href="conceptnum_1_1symmetric__preconditioner.html">symmetric_preconditioner</a> &ndash; <a href="conceptnum_1_1spd__preconditioner.html">spd_preconditioner</a></span></div><div class="kidx-group"><span class="kidx-title">Discrete structures</span><br/><span class="kidx-syms"><a href="conceptnum_1_1concepts_1_1equivalence__relation.html">equivalence_relation</a> &ndash; <a href="conceptnum_1_1concepts_1_1incidence__structure.html">incidence_structure</a> &ndash; <a href="conceptnum_1_1concepts_1_1weighted__incidence.html">weighted_incidence</a> &ndash; <a href="conceptnum_1_1concepts_1_1addressable__priority__queue.html">addressable_priority_queue</a> &ndash; <a href="conceptnum_1_1linear_1_1laplacian__graph.html">laplacian_graph</a></span></div><div class="kidx-group"><span class="kidx-title">Index spaces</span><br/><span class="kidx-syms"><a href="conceptnum_1_1square__extent__2d.html">square_extent_2d</a> &ndash; <a href="conceptnum_1_1cartesian__index__space__2d.html">cartesian_index_space_2d</a> &ndash; <a href="conceptnum_1_1periodic__neighbourhood__2d.html">periodic_neighbourhood_2d</a></span></div><div class="kidx-group"><span class="kidx-title">Fields and grids</span><br/><span class="kidx-syms"><a href="conceptnum_1_1structured__grid__2d.html">structured_grid_2d</a> &ndash; <a href="conceptnum_1_1scalar__field__like.html">scalar_field_like</a> &ndash; <a href="conceptnum_1_1solvable__field.html">solvable_field</a></span></div><div class="kidx-group"><span class="kidx-title">Ordinary differential equations</span><br/><span class="kidx-syms"><a href="conceptnum_1_1vec__field.html">vec_field</a> &ndash; <a href="conceptnum_1_1is__ode__problem.html">is_ode_problem</a> &ndash; <a href="conceptnum_1_1is__symplectic__ode__problem.html">is_symplectic_ode_problem</a> &ndash; <a href="conceptnum_1_1is__ode__stepper.html">is_ode_stepper</a></span></div><div class="kidx-group"><span class="kidx-title">Partial differential equations</span><br/><span class="kidx-syms"><a href="conceptnum_1_1grid__stencil.html">grid_stencil</a> &ndash; <a href="conceptnum_1_1assemblable__grid__operator.html">assemblable_grid_operator</a> &ndash; <a href="conceptnum_1_1implicit__step__operator.html">implicit_step_operator</a> &ndash; <a href="conceptnum_1_1field__stepper.html">field_stepper</a></span></div><div class="kidx-group"><span class="kidx-title">Spatial acceleration</span><br/><span class="kidx-syms"><a href="conceptnum_1_1position__accessor__2d.html">position_accessor_2d</a> &ndash; <a href="conceptnum_1_1neighbor__query__2d.html">neighbor_query_2d</a> &ndash; <a href="conceptnum_1_1smoothing__kernel.html">smoothing_kernel</a> &ndash; <a href="conceptnum_1_1periodic__lattice__2d.html">periodic_lattice_2d</a></span></div><div class="kidx-group"><span class="kidx-title">Stochastic</span><br/><span class="kidx-syms"><a href="conceptnum_1_1random__engine.html">random_engine</a> &ndash; <a href="conceptnum_1_1categorical__sampling.html">categorical_sampling</a> &ndash; <a href="conceptnum_1_1energy__difference.html">energy_difference</a></span></div><div class="kidx-group"><span class="kidx-title">Quadrature</span><br/><span class="kidx-syms"><a href="conceptnum_1_1quadrature__rule.html">quadrature_rule</a> &ndash; <a href="conceptnum_1_1contour__rule.html">contour_rule</a></span></div><div class="kidx-group"><span class="kidx-title">Spectral</span><br/><span class="kidx-syms"><a href="conceptnum_1_1transform__plan.html">transform_plan</a> &ndash; <a href="conceptnum_1_1unitary__transform.html">unitary_transform</a></span></div><div class="kidx-group"><span class="kidx-title">Statistics</span><br/><span class="kidx-syms"><a href="conceptnum_1_1streaming__accumulator.html">streaming_accumulator</a> &ndash; <a href="conceptnum_1_1moment__accumulator.html">moment_accumulator</a></span></div><div class="kidx-group"><span class="kidx-title">Root finding</span><br/><span class="kidx-syms"><a href="conceptnum_1_1bracketable__function.html">bracketable_function</a></span></div><div class="kidx-group"><span class="kidx-title">Problem dispatch</span><br/><span class="kidx-syms"><a href="conceptnum_1_1is__explicit__ode__alg.html">is_explicit_ode_alg</a> &ndash; <a href="conceptnum_1_1is__mcmc__alg.html">is_mcmc_alg</a></span></div>
</div>

---

## 1. The hierarchy

Each concept is the one above it equipped with one more operation, or carrying one more law.

```
field<T>                  + - * /, 0, 1
 └ additive_group<V>      dimension, zero_like, copy
    └ vector_space<V>     + scale, axpy                  ── law::vector_space
       └ normed_space<V>  + norm                         ── law::normed_space
          └ inner_product_space<V>  + inner              ── law::inner_product_space
             └ hilbert_space<V>     ‖x‖² = ⟨x,x⟩         ── law::hilbert_space

linear_map<Op>            domain and codomain are spaces ── law::linear_map
 └ linear_operator<Op>    + apply, rows, cols
    └ endomorphism<Op>    domain == codomain             ── law::endomorphism
       └ normal_operator          AA* = A*A              ── law::normal
          ├ self_adjoint_operator     A = A*             ── law::self_adjoint
          │  └ psd_operator             ⟨x,Ax⟩ ≥ 0       ── law::psd
          │     ├ spd_operator            ⟨x,Ax⟩ > 0     ── law::spd
          │     └ projection_operator     P = P* = P²    ── law::projection
          ├ skew_adjoint_operator     A = −A*            ── law::skew_adjoint
          └ unitary_operator          A*A = I            ── law::unitary
```

The stronger concept implies every weaker one, with nothing restated:

```cpp
// num::vec carries an inner product, so it satisfies everything below that.
static_assert(num::hilbert_space<num::vec>);
static_assert(num::inner_product_space<num::vec>);
static_assert(num::normed_space<num::vec>);
static_assert(num::vector_space<num::vec>);
static_assert(num::additive_group<num::vec>);

// One law in, every weaker concept out.
num::mat A = num::identity(4);
auto spd   = num::operators::assume_spd(num::operators::dense_op(A));
static_assert(num::spd_operator<decltype(spd)>);
static_assert(num::psd_operator<decltype(spd)>);           // implied
static_assert(num::self_adjoint_operator<decltype(spd)>);  // implied
static_assert(num::normal_operator<decltype(spd)>);        // implied
static_assert(num::linear_operator<decltype(spd)>);        // implied
```

The laws are partially ordered by implication. That order has meets but no joins.

```cpp
namespace L = num::law;

// A projector is positive semidefinite: ⟨x,Px⟩ = ⟨x,P²x⟩ = ‖Px‖² ≥ 0.
static_assert(std::derived_from<L::projection, L::psd>);

// Every pair has a greatest lower bound...
static_assert(std::same_as<L::meet_t<L::spd, L::psd>,     L::psd>);
static_assert(std::same_as<L::meet_t<L::spd, L::unitary>, L::normal>);

// ...but nothing implies both spd and unitary, so there is no least upper bound.
// The structure is a partial order, not a lattice.
```

Storage layout is described separately, under `num::repr`. Bandedness is a statement about
memory, not about a linear map.

```cpp
static_assert(num::repr::contiguous<num::vec>);
static_assert(num::repr::dense_row_major<num::mat>);
static_assert(num::repr::csr<num::spmat>);
```

---

## 2. Declaring what a type satisfies

A type you own declares its laws with a member typedef. Implication supplies the rest.

```cpp
struct custom_1d_laplacian {
    using math_laws     = num::math::type_list<num::law::spd>;  // the law it claims
    using domain_type   = num::vec;                             // the spaces it maps between
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

static_assert(num::spd_operator<custom_1d_laplacian>);
```

For a type you do not control, attach the claim from outside:

```cpp
namespace num::math {
template <> struct claims_of<third_party_matrix> {
    using type = type_list<law::self_adjoint>;
};
}
```

A type never acquires a law from its syntax. `std::string` defines `operator+` and is not a
vector space.

---

## 3. Laws survive operations that preserve them

The rule for a sum is the meet: \f$A + B\f$ satisfies whatever both operands satisfy, since
\f$\langle x, (A+B)x \rangle = \langle x, Ax \rangle + \langle x, Bx \rangle\f$.

```cpp
num::mat M = num::identity(4);
num::mat N = num::identity(4);
auto a = num::operators::assume_spd(num::operators::dense_op(M));
auto b = num::operators::assume_spd(num::operators::dense_op(N));

auto s = num::operators::sum(a, b);
static_assert(num::spd_operator<decltype(s)>);   // derived at compile time, no probe

num::vec rhs(4, 1.0), x(4, 0.0);
num::cg(s, rhs, x, 1e-12, 100);                  // accepted with nothing re-asserted
```

It derives exactly the meet and never more:

```cpp
num::mat M = num::identity(4);
auto spd  = num::operators::assume_spd(num::operators::dense_op(M));
auto sym  = num::operators::assume_symmetric(num::operators::dense_op(M));
auto uni  = num::operators::assume_orthogonal(num::operators::dense_op(M));
auto bare = num::operators::dense_op(M);

auto with_sym = num::operators::sum(spd, sym);
static_assert( num::self_adjoint_operator<decltype(with_sym)>);
static_assert(!num::spd_operator<decltype(with_sym)>);        // sym may be indefinite

auto with_uni = num::operators::sum(spd, uni);
static_assert( num::normal_operator<decltype(with_uni)>);
static_assert(!num::self_adjoint_operator<decltype(with_uni)>);

auto with_bare = num::operators::sum(spd, bare);
static_assert(!num::self_adjoint_operator<decltype(with_bare)>);  // nothing in, nothing out
```

Projection carries the law onto the subspace. \f$PA\f$ is not self-adjoint globally, since
\f$(PA)^* = AP \neq PA\f$. But \f$PAx = PAPx\f$ for every \f$x \in S\f$, and \f$PAP\f$
inherits definiteness from \f$A\f$ — which is what a `_on<S>` law asserts.

```cpp
num::mat M = num::identity(4);
auto a  = num::operators::assume_spd(num::operators::dense_op(M));
auto pa = num::operators::projected(a, num::space::zero_sum{});

static_assert( num::claims<decltype(pa), num::law::spd_on<num::space::zero_sum>>);
static_assert(!num::claims<decltype(pa), num::law::spd>);           // correctly refused
static_assert(!num::claims<decltype(pa), num::law::self_adjoint>);  // correctly refused
```

Only unconditional theorems are derived. A congruence \f$P^\top A P\f$ preserves
definiteness only when \f$P\f$ has full column rank, and a product of self-adjoint
operators is self-adjoint only when they commute. Neither side condition is checkable, so
both stay explicit assumptions.

---

## 4. What the compiler says

Output below is from real compiler and program runs, trimmed to the lines that identify the
cause.

### No law claimed

```cpp
// DOES NOT COMPILE -- the diagnostic is below.
num::mat A(3, 3, 1.0);
num::vec b(3, 1.0), x(3, 0.0);
num::cg(num::operators::dense_op(A), b, x, 1e-10, 100);
```

```
error: no matching function for call to 'cg'
cg.hpp:151: note: because 'claims<num::operators::dense_op, law::spd>' evaluated to false
```

### Law too weak

```cpp
// DOES NOT COMPILE -- the diagnostic is below.
auto sym = num::operators::assume_symmetric(num::operators::dense_op(A));
num::cg(sym, b, x, 1e-10, 100);
```

```
error: no matching function for call to 'cg'
cg.hpp:151: note: because 'claims<structured_op<dense_op, law::self_adjoint>, law::spd>'
                  evaluated to false
```

The wrapper carries its law in its type, so the message shows what was claimed beside what
was required. `num::minres` accepts this operator; `num::cg` does not.

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

The notes descend the hierarchy to the cause.

### Spaces not declared

```cpp
// DOES NOT COMPILE -- the diagnostic is below.
struct my_op {
    using math_laws = num::math::type_list<num::law::spd>;
    // no domain_type / codomain_type
};
static_assert(num::spd_operator<my_op>);
```

```
note: because 'psd_operator<my_op, void, void>' evaluated to false
```

`void, void` is the tell: the spaces defaulted to nothing.

### A claim that is false

The compiler checks that a law was *claimed*. Two runtime layers check that it is *true*.

```cpp
// Compiles; throws at run time. The output is below.
num::mat A(3, 3, 0.0);
A(0,0) = 2; A(1,1) = 2; A(2,2) = 2; A(0,1) = 5.0; A(1,0) = -5.0;   // not symmetric
auto spd = num::operators::assume_spd(num::operators::dense_op(A));
```

```
[PropertyError] Error at example.cpp:7 in int main():
  assume_symmetric() assertion failed: relative |<x,Ay> - conj(<y,Ax>)| = 1.649990
  on probe 0 exceeds tolerance 0.000000, so the operator is NOT self-adjoint.
```

`assume_spd` verifies the whole chain beneath `law::spd`, weakest first, so the
*self-adjointness* probe fires before definiteness is ever considered.

Sampling can miss a violation. The algorithm then catches it:

```
caught: cg: positive-definite curvature invariant was violated
```

That check costs \f$\mathcal{O}(1)\f$ per iteration against \f$\mathcal{O}(n)\f$ of work,
and `NDEBUG` does not remove it.

---

## 5. Attaching evidence to a value

A law describes a type. Evidence describes a particular matrix.

```cpp
num::mat A = num::identity(3);

auto claimed   = num::assume_spd(A);              // claim; sampled under the active preset
auto validated = num::make_spd(A);                // Cholesky, O(n^3); throws on failure
auto by_law    = num::assume<num::law::spd>(A);   // claim, stated as a law
auto checked   = num::require<num::law::spd>(A);  // exhaustive, where a validator exists
```

`assume` records `evidence_origin::assumed`; `require` records `verified`. Both store the
source location the diagnostics report.

| Matrix tagger | Operator tagger | Invariant | Cost |
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
and reports failure through its return value.

```cpp
num::mat indefinite(2, 2, 0.0);
indefinite(0, 0) =  1.0;
indefinite(1, 1) = -1.0;

auto factor = num::unsafe::cholesky(indefinite);   // factor.success == false, no throw
```

Available: `num::unsafe::cholesky`, `num::unsafe::lu`, `num::unsafe::eig_sym`,
`num::unsafe::cg`, `num::unsafe::lanczos`.

---

## 7. Diagnostic presets

`NUMERICS_DIAGNOSTICS` decides at compile time what checking code exists. The runtime preset
decides whether it runs.

| `NUMERICS_DIAGNOSTICS` | Contains | Default in |
| :--- | :--- | :--- |
| `0` | nothing; every check and probe discarded | — |
| `1` | shape checks: dimensions, emptiness, finiteness | builds with `NDEBUG` |
| `2` | property sampling as well | builds without `NDEBUG` |

Sampling costs \f$\mathcal{O}(n^2)\f$, so it is not a Release default:

```
ceiling=1   assume_spd =  0.00 ms   cg = 3.38 ms
ceiling=2   assume_spd = 31.19 ms   cg = 3.39 ms
```

At ceiling 1 no `verify_*_sample` symbol is emitted at all. Build with
`-DNUMERICS_DIAGNOSTICS=2` to keep sampling in an optimized build; this library's own test
suite does.

```cpp
num::set_preset(num::preset::strict);      // sample every property
num::set_preset(num::preset::balanced);    // shape and dimension checks only
num::set_preset(num::preset::production);  // everything off

// A request above the ceiling is clamped, and says so rather than downgrading silently.
num::set_preset(num::preset::strict);
if (!num::preset_fully_applied()) {
    // built below the requested level; rebuild with -DNUMERICS_DIAGNOSTICS=2
}

{
    num::scoped_preset guard(num::preset::production);
    // probing skipped in here; the previous preset returns at the closing brace
}
```

---

## 8. Axiom verification suites

These test a type against the axioms of a space over random elements. Use them on a type
you are adding to the hierarchy.

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

* @ref page_kernel "num::kernel" — the computational half of the library
