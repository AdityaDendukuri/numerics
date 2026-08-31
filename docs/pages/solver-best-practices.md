# Solver Best Practices {#page_solver_best_practices}

Select the weakest solver whose mathematical requirements match the system:

\f[
A x = b
\f]

---

## 1. Linear System Matrix Classes

| Operator / Matrix Class | Condition | Direct Solver | Iterative Solver |
| :--- | :--- | :--- | :--- |
| **Symmetric Positive Definite** | \f$A = A^T, \ x^T A x > 0\f$ | `num::cholesky` | `num::cg`, `num::pcg` |
| **Symmetric Indefinite** | \f$A = A^T\f$ | `num::lu`, `num::qr` | `num::minres` |
| **General Square** | No symmetry | `num::lu` | `num::gmres`, `num::bicgstab` |
| **Rectangular / Least Squares** | \f$\min_x \Vert A x - b \Vert_2\f$ | `num::qr_solve`, `num::svd` | `num::lsqr` |

---

## 2. Invariant Propagation

Prefer constructors that establish mathematical structure in the returned type:

```cpp
// 1. Backward Euler discretization is SPD by construction:
auto A = num::pde::backward_euler_operator(grid, coeff);
num::Vector rhs(grid.size(), 1.0), x(grid.size(), 0.0);
num::SolverResult info = num::cg(A, rhs, x); // Accepted directly without assume_spd
```

```cpp
// 2. External assembly requires explicit assertion or validation:
num::SparseMatrix A_sp = assemble_spd_matrix();
num::operators::SparseOp op(A_sp);

auto spd = num::operators::assume_spd(op); // Assertion tag
num::SolverResult info = num::cg(spd, b, x);
```

---

## 3. Preconditioning

Use `num::pcg` for ill-conditioned SPD systems when CG iteration count is high:

```cpp
num::SparseMatrix A = assemble_spd_matrix();
num::operators::SparseOp Aop(A);

auto M = num::jacobi_preconditioner(A); // M represents M^{-1} action
num::SolverResult info = num::pcg(num::operators::assume_spd(Aop), M, b, x);
```

---

## 4. Matrix-Free Operator Selection

* For self-adjoint stencils (diffusion, elliptic operators): `num::cg` with `assume_spd` or `num::minres` with `assume_symmetric`.
* For nonsymmetric operators (advection, Jacobian-free Newton–Krylov, upwind schemes): `num::gmres`.

```cpp
auto J = num::operators::make_op(apply_jacobian, n);
num::gmres(J, rhs, x, /*tol=*/1e-8, /*max_iter=*/1000, /*restart=*/40);
```

