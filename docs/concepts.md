---
scope: num::math
---

# Concepts, Laws & Diagnostics

Concepts express two kinds of requirement. **Structure** is decided by the compiler from the
operations a type provides. **Laws** are properties the compiler cannot decide, so a type
declares them and a probe samples them.

```cpp
// Structure: the operations settle it.
static_assert( num::vector_space<num::vec<num::real>>);
static_assert( num::vector_space<std::vector<double>>);
static_assert(!num::vector_space<std::vector<int>>);   // int is not a field

// A law: the caller states it.
num::mat<num::real> A = num::identity(4);
auto op  = num::operators::dense_op(A);        // a linear operator, claiming nothing
auto sym = num::assume_symmetric(op);          // now claims law::self_adjoint
static_assert(!num::self_adjoint_operator<decltype(op)>);
static_assert( num::self_adjoint_operator<decltype(sym)>);
```

---

Every concept, grouped by what it describes, is listed under [All concepts](reference/concepts.md).

---

## 1. The hierarchy

```
field<T>                         floating point, real or complex
vector_space<V>                  dimension, zero_like, scale, axpy over a field
 └ inner_product_space<V>        + inner, norm
linear_operator<Op, X, Y>        apply, rows, cols between vector spaces
 └ self_adjoint_operator<Op, V>  + claims law::self_adjoint
    └ psd_operator<Op, V>        + claims law::psd
       └ spd_operator<Op, V>     + claims law::spd
```

A space is anything the operations `num::math::dimension`, `zero_like`, `scale`, `axpy`,
`inner` and `norm` accept. Each operation calls a `tag_invoke` overload when the type has
one, and otherwise loops over `v[i]`. So `std::vector<double>` is an inner product space
with no declaration.

Linearity has no law. It is a precondition of `linear_operator`, in the way `std::regular`
states requirements the compiler cannot check. A law exists only where an algorithm depends
on it:

| law | implies | required by |
| :--- | :--- | :--- |
| `law::self_adjoint` | | `minres`, `lanczos`, `eig_sym` |
| `law::psd` | `self_adjoint` | graph Laplacians, before a subspace restriction |
| `law::spd` | `psd` | `cg`, `pcg`, `cholesky`, `sqrt_lanczos` |
| `law::diagonally_dominant` | | `jacobi`, `gauss_seidel` |
| `law::spd_on<S>` | `psd_on<S>`, `self_adjoint_on<S>` | `pcg` on the subspace `S` |

A stronger law derives from the weaker ones, so one claim satisfies every weaker concept:

```cpp
auto spd = num::assume_spd(num::operators::dense_op(A));
static_assert(num::spd_operator<decltype(spd)>);
static_assert(num::psd_operator<decltype(spd)>);
static_assert(num::self_adjoint_operator<decltype(spd)>);
```

Storage layout is described separately, under `num::repr`. Bandedness is a statement about
memory, not about a linear map.

```cpp
static_assert(num::repr::contiguous<num::vec<num::real>>);
static_assert(num::repr::dense_row_major<num::mat<num::real>>);
static_assert(num::repr::csr<num::spmat>);
```

---

## 2. Declaring a law

A type declares the laws it satisfies with a member alias:

```cpp
struct custom_1d_laplacian {
    using laws          = num::law::list<num::law::spd>;
    using domain_type   = num::vec<num::real>;
    using codomain_type = num::vec<num::real>;

    num::idx n;
    [[nodiscard]] num::idx rows() const noexcept { return n; }
    [[nodiscard]] num::idx cols() const noexcept { return n; }

    void apply(const num::vec<num::real> &x, num::vec<num::real> &y) const {
        for (num::idx i = 0; i < n; ++i) {
            y[i] = 2.0 * x[i] - (i > 0 ? x[i - 1] : 0.0) - (i + 1 < n ? x[i + 1] : 0.0);
        }
    }
};

static_assert(num::spd_operator<custom_1d_laplacian>);
```

A type may declare several laws, including incomparable ones such as `spd` and
`diagonally_dominant`. `num::claims<T, L>` is true when `T` declares `L` or a law that
implies it.

A value whose type you do not control gets a law from `num::assume` instead; see section 4.

### A declared law must be impossible to violate

A declaration is a promise about every instance, and nothing checks it. It is sound only
when no violating instance can be built. Every declaration in the library earns its law in
one of three ways.

A constructor can reject the bad inputs:

```cpp
// SPD because a non-positive diagonal cannot get past the constructor.
class positive_diagonal {
  public:
    using laws = num::law::list<num::law::spd>;

    explicit positive_diagonal(num::vec<num::real> d) : d_(std::move(d)) {
        for (const num::real value : d_) {
            if (!(value > 0.0) || !std::isfinite(value)) {
                throw std::invalid_argument("diagonal must be positive and finite");
            }
        }
    }
  private:
    num::vec<num::real> d_;
};
```

A constraint can reject the bad operands:

```cpp
// p(A) is positive definite only when A is.
template <class Op>
requires num::spd_operator<Op>
class chebyshev_preconditioner final { /* ... */ };
```

The structure of `apply` can leave nothing to violate. A symmetric stencil is self-adjoint
whatever its coefficients.

| type | claims | earned by |
| :--- | :--- | :--- |
| `jacobi_preconditioner` | `spd` | constructor: every entry positive and finite |
| `chebyshev_preconditioner<Op>` | `spd` | `requires spd_operator<Op>` |
| `backward_euler_2d` | `spd` | constructor: `coeff >= 0`, so Gershgorin applies |
| `backward_euler_operator_2d` | `spd` | the same check |
| `laplacian_2d` | `self_adjoint` | structure: symmetric 5-point stencil |

If a program can construct a violating instance, the law belongs behind `num::assume`, where
the caller takes responsibility and the diagnostics sample it.

---

## 3. Laws that follow from an operation

$P_S A$ is not self-adjoint, since $(P_S A)^* = A P_S \neq P_S A$. But $P_S A x = P_S
A P_S x$ for $x \in S$, and $P_S A P_S$ keeps the law of $A$ on $S$.
`num::operators::projected` carries that law over, so a graph Laplacian can be solved on the
zero-sum subspace without a second assertion.

```cpp
auto a  = num::assume_spd(num::operators::dense_op(M));
auto pa = num::operators::projected(a, num::space::zero_sum{});

static_assert( num::claims<decltype(pa), num::law::spd_on<num::space::zero_sum>>);
static_assert(!num::claims<decltype(pa), num::law::spd>);
static_assert(!num::claims<decltype(pa), num::law::self_adjoint>);
```

A weaker operand gives a weaker restriction, and an operand claiming nothing gives nothing.

Strict diagonal dominance and definiteness are incomparable. $\begin{pmatrix} 1 & 0.9 \\
0.9 & 1\end{pmatrix}$ is SPD and not dominant, and a dominant matrix need not be symmetric.
A routine that accepts either says so with `||`:

```cpp
// gauss_seidel converges for strictly diagonally dominant or SPD A.
num::gauss_seidel(num::assume_diagonally_dominant(A), b, x);
num::gauss_seidel(num::assume_spd(A), b, y);
```

---

## 4. Attaching a law to a value

`num::with_law<T, L>` owns a matrix or operator `T` and claims `L`. It forwards the shape, the
entries and the action. A value with a stronger law converts to one with a weaker law, so an
SPD matrix is accepted where a symmetric one is required.

```cpp
num::mat<num::real> A = num::identity(3);

auto claimed  = num::assume_spd(A);              // sampled under the active preset
auto by_law   = num::assume<num::law::spd>(A);   // the same, naming the law
auto verified = num::make_spd(A);                // Cholesky, O(n^3); throws on failure

num::eig_sym(verified);                          // spd converts to self_adjoint
```

| attach | law | checked by | cost |
| :--- | :--- | :--- | :--- |
| `assume_symmetric(A)` | `self_adjoint` | sampled $\langle x, Ay\rangle = \overline{\langle y, Ax\rangle}$ | $\mathcal{O}(n^2)$ |
| `assume_psd(A)` | `psd` | sampled $\langle x, Ax\rangle \ge 0$ | $\mathcal{O}(n^2)$ |
| `assume_spd(A)` | `spd` | sampled, plus a power-iteration bound on $\lambda_{\min}$ | $\mathcal{O}(n^2)$ |
| `assume_diagonally_dominant(A)` | `diagonally_dominant` | every row, exactly | $\mathcal{O}(n^2)$ |
| `make_symmetric(A)` | `self_adjoint` | every entry, exactly | $\mathcal{O}(n^2)$ |
| `make_spd(A)` | `spd` | Cholesky, exactly | $\mathcal{O}(n^3)$ |

Every `assume` samples linearity and checks squareness as well. Squareness is checked in
every build, since it costs nothing.

Constructing `with_law<T, L>(value)` directly attaches the law with no check. That is for
code that has established the law by other means.

---

## 5. What the compiler says

The output below is from real compiler runs, trimmed to the lines that identify the cause.

### No law claimed

```cpp
// DOES NOT COMPILE
num::cg(num::operators::dense_op(A), b, x, 1e-10, 100);
```

```
error: no matching function for call to 'cg'
note: because 'math::spd_operator<num::operators::dense_op, num::vec<double>>' evaluated to false
note: because 'psd_operator<num::operators::dense_op, num::vec<double>>' evaluated to false
note: because 'self_adjoint_operator<num::operators::dense_op, num::vec<double>>' evaluated to false
note: because 'claims<num::operators::dense_op, law::self_adjoint>' evaluated to false
```

### Law too weak

```cpp
// DOES NOT COMPILE
auto sym = num::assume_symmetric(num::operators::dense_op(A));
num::cg(sym, b, x, {.tolerance = 1e-10, .max_iterations = 100});
```

```
error: no matching function for call to 'cg'
note: because 'claims<num::with_law<num::operators::dense_op, num::law::self_adjoint>, law::psd>'
      evaluated to false
```

The claimed law is in the type, so the message shows it beside the one required. `num::minres`
accepts this operator.

### Type outside the hierarchy

```cpp
// DOES NOT COMPILE
static_assert(num::vector_space<std::vector<int>>);
```

```
error: static assertion failed
note: because 'std::vector<int>' does not satisfy 'vector_space'
note: because 'scalar_t<std::vector<int>>' (aka 'int') does not satisfy 'field'
```

### Spaces not declared

```cpp
// DOES NOT COMPILE
struct my_op {
    using laws = num::law::list<num::law::spd>;
    // no domain_type or codomain_type
    /* rows, cols, apply */
};
static_assert(num::spd_operator<my_op>);
```

```
note: because 'linear_operator<my_op, void, void>' evaluated to false
note: because 'void' does not satisfy 'vector_space'
```

`void, void` means the spaces were never named.

### A claim that is false

The compiler checks that a law was claimed. Two runtime layers check that it is true.

```cpp
// Compiles; throws at run time.
num::mat<num::real> A(3, 3, 0.0);
A(0,0) = 2; A(1,1) = 2; A(2,2) = 2; A(0,1) = 5.0; A(1,0) = -5.0;   // not symmetric
auto spd = num::assume_spd(num::operators::dense_op(A));
```

```
[PropertyError] Error at example.cpp:7 in int main():
  assume_symmetric() assertion failed: relative |<x,Ay> - conj(<y,Ax>)| = 1.649990
  on probe 0 exceeds tolerance 0.000000, so the operator is NOT self-adjoint.
```

`assume_spd` checks the weaker laws first, so the self-adjointness probe fires before
definiteness is considered. Sampling can miss a violation. The algorithm then catches it:

```
caught: cg: positive-definite curvature invariant was violated
```

That check costs $\mathcal{O}(1)$ per iteration and stays in `NDEBUG` builds.

---

## 6. Bypassing enforcement

Each law-gated routine has a counterpart under `num::unsafe` that takes a plain matrix and
reports failure through its return value.

```cpp
num::mat<num::real> indefinite(2, 2, 0.0);
indefinite(0, 0) =  1.0;
indefinite(1, 1) = -1.0;

auto factor = num::unsafe::cholesky(indefinite);   // factor.success == false, no throw
```

Available: `num::unsafe::cholesky`, `num::unsafe::eig_sym`, `num::unsafe::cg`,
`num::unsafe::lanczos`.

---

## 7. Diagnostic presets

`NUMERICS_DIAGNOSTICS` decides at compile time what checking code exists. The runtime preset
decides whether it runs.

| `NUMERICS_DIAGNOSTICS` | Contains | Default in |
| :--- | :--- | :--- |
| `0` | nothing | |
| `1` | shape checks: dimensions, emptiness, finiteness | builds with `NDEBUG` |
| `2` | property sampling as well | builds without `NDEBUG` |

Sampling costs $\mathcal{O}(n^2)$, so it is not a Release default:

```
ceiling=1   assume_spd =  0.00 ms   cg = 3.38 ms
ceiling=2   assume_spd = 31.19 ms   cg = 3.39 ms
```

Build with `-DNUMERICS_DIAGNOSTICS=2` to keep sampling in an optimized build. This library's
own test suite does.

```cpp
num::set_preset(num::preset::strict);      // sample every property
num::set_preset(num::preset::balanced);    // shape and dimension checks only
num::set_preset(num::preset::production);  // everything off

// A request above the ceiling is clamped, and reports it.
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

## 8. Requirements by routine

| Routine | Requires | Alternative for weaker input |
| :--- | :--- | :--- |
| `num::cholesky` | a dense matrix claiming `law::spd` | `num::lu` |
| `num::cg` | `num::spd_operator` | `num::minres`, `num::gmres` |
| `num::pcg` | `num::spd_operator` for operator and preconditioner | `num::gmres` |
| `num::minres` | `num::self_adjoint_operator` | `num::gmres` |
| `num::gmres` | `num::linear_operator` | |
| `num::lu` | a square matrix, checked at run time | `num::qr` |
| `num::eig_sym` | a dense matrix claiming `law::self_adjoint` | `num::power_iteration` |
| `num::lanczos` | `num::self_adjoint_operator` | `num::power_iteration` |
| `num::jacobi` | a matrix claiming `law::diagonally_dominant` | `num::gauss_seidel` |
| `num::gauss_seidel` | `law::diagonally_dominant` or `law::spd` | |

---

## Example

See the example program [14_concepts_and_property_invariants.cpp](reference/examples/14_concepts_and_property_invariants.md).

---

## See also

* [num::kernel](reference/kernel.md), the computational half of the library
