# Unified Problem Dispatch {#page_solve}

The `solve` module provides problem descriptors and algorithm tags, so a problem is stated once and handed to whichever method suits it.

A descriptor names what is being solved. An algorithm tag names how, and carries that method's tolerances and options. The pair goes to `num::solve`.

---

## 1. Linear Systems

\f[ A \mathbf{x} = \mathbf{b} \f]

```cpp
num::Matrix A = make_spd_matrix();
num::Vector b{1.0, 2.0, 3.0};

num::operators::DenseOp Aop(A);
auto spd = num::operators::assume_spd(Aop);

auto solution = num::solve(num::LinearProblem{spd, b},
                           num::CG{.tol = 1e-10, .max_iter = 500});

// solution.u, solution.iterations, solution.residual, solution.converged
```

`LinearProblem` is a non-owning view over \f$A\f$ and \f$\mathbf{b}\f$, so it is bound at the call site rather than stored.

The algorithm tag is checked against the operator. `num::CG` requires the SPD property and `num::MINRES` requires self-adjointness, so pairing a tag with an operator that lacks the property is a compile error rather than a silent failure to converge:

```cpp
auto sym = num::operators::assume_symmetric(Aop);

num::solve(num::LinearProblem{sym, b}, num::MINRES{});   // Self-adjoint is enough.
num::solve(num::LinearProblem{sym, b}, num::CG{});       // error: CG requires SPD.
num::solve(num::LinearProblem{Aop, b}, num::GMRES{});    // GMRES requires only linearity.
```

### Available algorithms

| Tag | Requires | Options |
| :--- | :--- | :--- |
| `num::CG` | SPD operator | `tol`, `max_iter`, `backend` |
| `num::PCG<M>` | SPD operator and preconditioner | `tol`, `max_iter`, `backend` |
| `num::MINRES` | Self-adjoint operator | `tol`, `max_iter`, `backend` |
| `num::GMRES` | Linear operator | `tol`, `max_iter`, `restart`, `backend` |

---

## 2. Reusing a Factorization

A cache holds the setup so repeated right-hand sides skip it:

```cpp
auto cache = num::init(num::LinearProblem{spd, b}, num::CG{});

for (const auto &rhs : right_hand_sides) {
    auto solution = num::solve(cache); // Reuses the operator and preconditioner setup.
}
```

---

## 3. Initial Value Problems

\f[ \dot{\mathbf{u}} = f(t, \mathbf{u}), \qquad \mathbf{u}(t_0) = \mathbf{u}_0 \f]

```cpp
num::ODEProblem problem{
    .f = [](num::real t, const num::Vector &u, num::Vector &dudt) { dudt[0] = -u[0]; },
    .u0 = num::Vector{1.0},
    .t0 = 0.0,
    .tf = 1.0,
};

auto result = num::solve(problem, num::RK45{.h = 1e-3, .rtol = 1e-8, .atol = 1e-10});
```

### Available integrators

| Tag | Order | Options |
| :--- | :--- | :--- |
| `num::Euler` | 1 | `h` |
| `num::RK4` | 4 | `h` |
| `num::RK45` | 5, embedded 4 | `h`, `rtol`, `atol`, `max_steps` |

An observer runs after each accepted step:

```cpp
auto result = num::solve(problem, num::RK45{}, [](num::real t, const num::Vector &u) {
    record(t, u[0]);
});
```

---

## 4. Compile-Time Concepts

```cpp
static_assert(num::IsExplicitODEAlg<num::RK45>);
static_assert(num::IsMCMCAlg<num::Metropolis>);
```

For how the property hierarchy gates the linear solvers, see @ref page_concepts. For choosing between methods, see @ref page_solver_best_practices.
