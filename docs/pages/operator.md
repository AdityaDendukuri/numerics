# Matrix-Free Operators {#page_operator}

Non-owning linear operator wrappers and matrix-free callables implementing \f$y \leftarrow A x\f$.

---

## 1. Operator Types

### Dense Matrix Operator (num::operators::dense_op)
Non-owning adapter over `num::mat`.

```cpp
num::mat A(3, 3, 0.0);
num::operators::dense_op op(A); // Non-owning reference

num::vec x{1.0, 2.0, 3.0};
num::vec y(3, 0.0);
op.apply(x, y); // y <- A * x
```

### Sparse Matrix Operator (num::operators::sparse_op)
Non-owning adapter over `num::spmat` (CSR).

```cpp
num::spmat A = num::spmat::from_triplets(n, n, rows, cols, values);
num::operators::sparse_op op(A);

num::vec x(n, 1.0), y(n, 0.0);
op.apply(x, y); // y <- A * x
```

### Callable Custom Operator (num::operators::make_op)
Creates a matrix-free linear operator from a callable \f$(x, y) \to \text{void}\f$.

```cpp
auto laplacian = num::operators::make_op(
    [N](const num::vec& x, num::vec& y) {
        for (num::idx i = 0; i < N; ++i) {
            y[i] = 2.0 * x[i] - (i > 0 ? x[i - 1] : 0.0) - (i + 1 < N ? x[i + 1] : 0.0);
        }
    }, N);

static_assert(num::linear_operator<decltype(laplacian)>);
```

### Subspace Projection Operator (num::operators::projected)
Projects an operator \f$M\f$ onto a subspace \f$S\f$: \f$M_S = P_S M P_S\f$.

```cpp
num::space::zero_sum S;
auto proj_M = num::operators::projected(M, S);
```

---

## 2. Invariant Tagging on Operators

```cpp
// 1. Tag symmetry / self-adjointness
auto sym_op = num::operators::assume_symmetric(op);
static_assert(num::self_adjoint_operator<decltype(sym_op)>);

// 2. Tag positive definiteness (SPD)
auto spd_op = num::operators::assume_spd(op);
static_assert(num::spd_operator<decltype(spd_op)>);

// Solve with CG using tagged operator
num::vec b(n, 1.0), x(n, 0.0);
num::solver_result res = num::cg(spd_op, b, x, 1e-8, 500);
```

---

## 3. Custom Operator Interface

Any struct satisfying `num::linear_operator` can be passed to solvers without inheritance:

```cpp
struct Custom1DLaplacian {
    using math_laws = num::math::type_list<num::law::spd>; // the law it claims
    using domain_type = num::vec;                       // the spaces it maps between
    using codomain_type = num::vec;
    num::idx n;

    [[nodiscard]] num::idx rows() const noexcept { return n; }
    [[nodiscard]] num::idx cols() const noexcept { return n; }

    void apply(const num::vec& x, num::vec& y) const {
        for (num::idx i = 0; i < n; ++i) {
            y[i] = 2.0 * x[i] - (i > 0 ? x[i - 1] : 0.0) - (i + 1 < n ? x[i + 1] : 0.0);
        }
    }
};

static_assert(num::spd_operator<Custom1DLaplacian>);

Custom1DLaplacian L{100};
num::vec b(100, 1.0), x(100, 0.0);
num::cg(L, b, x); // Zero matrix assembly; accepted directly by CG
```

---

## Examples

- @example 02_iterative_krylov_solvers.cpp
- @example 10_banded_and_spd_operators.cpp
- @example 15_diffusion_evidence_cg.cpp


