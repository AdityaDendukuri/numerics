# Unified Problem Dispatch {#page_solve}

Façade interface: descriptors state the problem, algorithm tags state the method.

---

## 1. Linear Problems (num::linear_problem)

\f[
A \mathbf{x} = \mathbf{b}
\f]

```cpp
#include <numerics.hpp>

num::mat A = make_spd_matrix();
num::vec b{1.0, 2.0, 3.0};

num::operators::dense_op op(A);
auto spd = num::operators::assume_spd(op);

// Dispatch to CG
auto solution = num::solve(
    num::linear_problem{spd, b},
    num::cg_method{.tol = 1e-10, .max_iter = 500});

// solution.u, solution.iterations, solution.residual, solution.converged
```

### Algorithm Tags

| Tag | Requires | Options |
| :--- | :--- | :--- |
| `num::cg_method` | SPD operator | `tol`, `max_iter`, `backend` |
| `num::pcg_method<M>` | SPD operator and preconditioner | `tol`, `max_iter`, `backend` |
| `num::minres_method` | Self-adjoint operator | `tol`, `max_iter`, `backend` |
| `num::gmres_method` | Linear operator | `tol`, `max_iter`, `restart`, `backend` |

---

## 2. Solver Setup Caching (num::init)

Reuses solver and preconditioner allocation across multiple right-hand sides:

```cpp
auto cache = num::init(num::linear_problem{spd, b}, num::cg_method{});

for (const auto& rhs : rhs_list) {
    auto solution = num::solve(cache, rhs);
}
```

---

## 3. Initial Value Problems (num::ode_problem)

\f[
\dot{\mathbf{u}} = \mathbf{f}(t, \mathbf{u}), \qquad \mathbf{u}(t_0) = \mathbf{u}_0
\f]

```cpp
num::ode_problem problem{
    .f  = [](num::real t, const num::vec& u, num::vec& du) { du[0] = -u[0]; },
    .u0 = num::vec{1.0},
    .t0 = 0.0,
    .tf = 1.0,
};

auto res = num::solve(problem, num::rk45_method{.h = 1e-3, .rtol = 1e-8, .atol = 1e-10});
```

### Available Integrator Tags

| Tag | Order | Options |
| :--- | :--- | :--- |
| `num::euler_method` | 1 | `h` |
| `num::rk4_method` | 4 | `h` |
| `num::rk45_method` | 5 (embedded 4) | `h`, `rtol`, `atol`, `max_steps` |

Observer step callback:
```cpp
auto res = num::solve(problem, num::rk45_method{}, [](num::real t, const num::vec& u) {
    // record(t, u[0]);
});
```

---

## 4. Concepts

```cpp
static_assert(num::is_explicit_ode_alg<num::rk45_method>);
static_assert(num::is_mcmc_alg<num::metropolis_method>);
```

