# Solver Best Practices {#page_solver_best_practices}

Use the weakest solver whose mathematical assumptions match the system.

## Linear System Classes

For a general linear system

\f[
    Ax=b ,
\f]

choose the method from the structure of \f$A\f$:

| Matrix/operator class | Condition | Routine |
| --- | --- | --- |
| SPD | \f$A=A^T,\; x^T A x>0\f$ | `cg`, `pcg`, `cholesky` |
| symmetric indefinite | \f$A=A^T\f$ | `minres` |
| nonsymmetric/general | no symmetry assumption | `gmres`, `lu` |
| rectangular/least squares | \f$\min_x \|Ax-b\|_2\f$ | `qr_solve`, `svd` |

Do not use CG as a general-purpose Krylov method. If symmetry or positive
definiteness is not known, use GMRES.

## Preserve Structure

Prefer constructors that keep mathematical structure in the type:

```cpp
auto A = num::pde::backward_euler_operator(grid, coeff);
num::SolverResult info = num::cg(A, rhs, x);
```

`backward_euler_operator` owns the assembled sparse matrix and carries the SPD
operator tag. The solver call is short because the PDE builder supplies the
structure.

When structure comes from outside the library, state it explicitly:

```cpp
num::SparseMatrix A = assemble_spd_matrix();
num::operators::SparseOp Aop(A);

auto Aspd = num::operators::assume_spd(Aop);
num::SolverResult info = num::cg(Aspd, b, x);
```

`assume_spd` is an unchecked mathematical cast. It should be used when the
discretization, assembly routine, or factorization guarantees the property.

## Matrix Properties

Stored dense matrices can also carry declared structure:

```cpp
num::Matrix A = assemble_dense_spd_matrix();
auto Aspd = num::linalg::make_spd(A);

auto F = num::cholesky(Aspd);
num::cholesky_solve(F, b, x);
```

`make_spd` checks symmetry and the Cholesky pivots before constructing the SPD
wrapper. Use `assume_spd` only when the construction, discretization, or prior
factorization already proves the property.

## Preconditioning

Use PCG for SPD systems when CG iteration counts are too high:

```cpp
num::SparseMatrix A = assemble_spd_matrix();
num::operators::SparseOp Aop(A);

auto M = num::jacobi_preconditioner(A);
num::SolverResult info =
    num::pcg(num::operators::assume_spd(Aop), M, b, x);
```

A preconditioner represents applying \f$M^{-1}r\f$, not forming
\f$M^{-1}\f$ explicitly. The preconditioner must be compatible with the solver
class: SPD-compatible preconditioners for PCG, general preconditioners for
general Krylov methods.

## Matrix-Free Operators

For matrix-free code, runtime SPD validation is generally not available. The
library cannot inspect all entries of \f$A\f$ because they are never assembled.
Use property wrappers only when the formula is known:

```cpp
auto L = num::operators::make_op(
    [N](const num::Vector& u, num::Vector& Lu) {
        apply_negative_laplacian(u, Lu, N);
    },
    N * N);

num::cg(num::operators::assume_spd(L), rhs, x);
```

For advection, Jacobians, upwind discretizations, or nonsymmetric
preconditioned systems, use GMRES:

```cpp
auto J = num::operators::make_op(apply_jacobian, n);
num::gmres(J, rhs, x, 1e-8, 1000, 40);
```

The unified `solve(problem, algorithm)` form is useful when the algorithm is
selected at the call site:

```cpp
auto Aspd = num::operators::assume_spd(L);
num::LinearSolution rc = num::solve(num::LinearProblem{Aspd, rhs}, num::CG{.tol = 1e-10});
num::LinearSolution rg =
    num::solve(num::LinearProblem{J, rhs}, num::GMRES{.tol = 1e-8, .restart = 40});
```

## Practical Rule

Start from the mathematical class of the operator:

```text
SPD                 -> Cholesky, CG, PCG
symmetric indefinite -> MINRES
general square       -> LU, GMRES
rectangular          -> QR, SVD
```

Then choose direct or iterative form from size, sparsity, and whether only the
action \f$y=Ax\f$ is available.
