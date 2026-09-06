# Stencil Operator Examples {#page_stencil_hof}

Stencil routines are useful both as direct updates and as matrix-free linear
operators.

## Apply a 2D Laplacian

```cpp
num::vec u(N * N, 0.0);
num::vec Lu(N * N, 0.0);

num::laplacian_stencil_2d(u, Lu, N);
```

## Matrix-Free Operator

```cpp
auto A = num::operators::make_op(
    [N](const num::vec& x, num::vec& y) {
        num::laplacian_stencil_2d(x, y, N);
        num::scale(y, -1.0);
    },
    N * N);

num::solver_result info =
    num::cg(num::operators::assume_spd(A), rhs, sol, 1e-8, 1000);
```

The operator represents

\f[
    y = -L_h x .
\f]

## Periodic Diffusion Step

```cpp
double coeff = kappa * dt / (h * h);
num::pde::diffusion_step_2d(u, N, coeff);
```

## Fourth-Order Dirichlet Laplacian

```cpp
num::vec Lu4(N * N, 0.0);
num::laplacian_stencil_2d_4th(u, Lu4, N);
```

Use matrix-free operators when the sparse matrix would be expensive to assemble
or when the stencil is applied many times inside a Krylov method.
