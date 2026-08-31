# Unified Problem Dispatch {#page_solve}

Façade interface: descriptors state the problem, algorithm tags state the method.

---

## 1. Linear Problems (num::LinearProblem)

\f[
A \mathbf{x} = \mathbf{b}
\f]

```cpp
#include <numerics.hpp>

num::Matrix A = make_spd_matrix();
num::Vector b{1.0, 2.0, 3.0};

num::operators::DenseOp op(A);
auto spd = num::operators::assume_spd(op);

// Dispatch to CG
auto solution = num::solve(
    num::LinearProblem{spd, b},
    num::CG{.tol = 1e-10, .max_iter = 500});

// solution.u, solution.iterations, solution.residual, solution.converged
```

### Algorithm Tags

| Tag | Requires | Options |
| :--- | :--- | :--- |
| `num::CG` | SPD operator | `tol`, `max_iter`, `backend` |
| `num::PCG<M>` | SPD operator and preconditioner | `tol`, `max_iter`, `backend` |
| `num::MINRES` | Self-adjoint operator | `tol`, `max_iter`, `backend` |
| `num::GMRES` | Linear operator | `tol`, `max_iter`, `restart`, `backend` |

---

## 2. Solver Setup Caching (num::init)

Reuses solver and preconditioner allocation across multiple right-hand sides:

```cpp
auto cache = num::init(num::LinearProblem{spd, b}, num::CG{});

for (const auto& rhs : rhs_list) {
    auto solution = num::solve(cache, rhs);
}
```

---

## 3. Initial Value Problems (num::ODEProblem)

\f[
\dot{\mathbf{u}} = \mathbf{f}(t, \mathbf{u}), \qquad \mathbf{u}(t_0) = \mathbf{u}_0
\f]

```cpp
num::ODEProblem problem{
    .f  = [](num::real t, const num::Vector& u, num::Vector& du) { du[0] = -u[0]; },
    .u0 = num::Vector{1.0},
    .t0 = 0.0,
    .tf = 1.0,
};

auto res = num::solve(problem, num::RK45{.h = 1e-3, .rtol = 1e-8, .atol = 1e-10});
```

### Available Integrator Tags

| Tag | Order | Options |
| :--- | :--- | :--- |
| `num::Euler` | 1 | `h` |
| `num::RK4` | 4 | `h` |
| `num::RK45` | 5 (embedded 4) | `h`, `rtol`, `atol`, `max_steps` |

Observer step callback:
```cpp
auto res = num::solve(problem, num::RK45{}, [](num::real t, const num::Vector& u) {
    // record(t, u[0]);
});
```

---

## 4. Concepts

```cpp
static_assert(num::IsExplicitODEAlg<num::RK45>);
static_assert(num::IsMCMCAlg<num::Metropolis>);
```

