# Solver Best Practices {#page_solver_best_practices}

Select the weakest solver whose mathematical requirements match the system:

\f[
A x = b
\f]

---

## 1. Linear System Matrix Classes

| Operator / mat Class | Condition | Direct Solver | Iterative Solver |
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
num::vec rhs(grid.size(), 1.0), x(grid.size(), 0.0);
num::solver_result info = num::cg(A, rhs, x); // Accepted directly without assume_spd
```

```cpp
// 2. External assembly requires explicit assertion or validation:
num::spmat A_sp = assemble_spd_matrix();
num::operators::sparse_op op(A_sp);

auto spd = num::assume_spd(op); // Assertion tag
num::solver_result info = num::cg(spd, b, x);
```

---

## 3. Preconditioning

Use `num::pcg` for ill-conditioned SPD systems when CG iteration count is high:

```cpp
num::spmat A = assemble_spd_matrix();
num::operators::sparse_op Aop(A);

auto M = num::make_jacobi_preconditioner(A); // M represents M^{-1} action
num::solver_result info = num::pcg(num::assume_spd(Aop), M, b, x);
```

### Chebyshev polynomial preconditioning

`num::chebyshev_preconditioner` approximates \f$A^{-1}\f$ by the degree-m polynomial that
minimizes \f$\max_{\lambda \in [\ell, h]} |1 - \lambda p(\lambda)|\f$. It needs only the
operator's action, so it is the one preconditioner here that works on a matrix-free operator
built with `num::operators::make_op`. It also uses no global reductions.

A degree-m polynomial improves the condition number by about a factor of m. That is not
competitive on a second-order elliptic operator, whose condition number grows with the mesh;
use ApproxChol for SDD systems, or algebraic multigrid. It fits three cases:

- matrix-free operators, which have no entries to factor;
- moderately conditioned systems, such as mass matrices, shifted or damped operators, and
  regularized least squares;
- smoothing the upper spectrum inside a multigrid cycle.

The polynomial is positive definite only when \f$0 < \ell \le \lambda_{\min}\f$ and
\f$\lambda_{\max} \le h\f$. Bounds that miss part of the spectrum make it indefinite, and PCG
then loses its monotone error. `num::estimate_largest_eigenvalue` finds \f$h\f$ by power
iteration. There is no equally cheap estimate of \f$\ell\f$. Take it from the problem's
physics, from a shift or regularization parameter, or from `num::lanczos`, not from an assumed
condition number.

---

## 4. Matrix-Free Operator Selection

* For self-adjoint stencils (diffusion, elliptic operators): `num::cg` with `assume_spd` or `num::minres` with `assume_symmetric`.
* For nonsymmetric operators (advection, Jacobian-free Newton–Krylov, upwind schemes): `num::gmres`.

```cpp
auto J = num::operators::make_op(apply_jacobian, n);
num::gmres(J, rhs, x, /*tol=*/1e-8, /*max_iter=*/1000, /*restart=*/40);
```

