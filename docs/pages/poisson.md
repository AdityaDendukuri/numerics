# Poisson Solver Example {#page_poisson}

`num::pde::poisson2d` solves the Dirichlet problem

\f[
    -\Delta u = f,\qquad u|_{\partial\Omega}=0
\f]

on an \f$N\times N\f$ interior grid by DST-I diagonalization.

## Solve on a Square Grid

```cpp
constexpr int N = 63;
constexpr double h = 1.0 / (N + 1);

num::Matrix f(N, N, 0.0);
for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
        double x = (i + 1) * h;
        double y = (j + 1) * h;
        f(i, j) = 2.0 * std::numbers::pi * std::numbers::pi
                * std::sin(std::numbers::pi * x)
                * std::sin(std::numbers::pi * y);
    }
}

num::Matrix u = num::pde::poisson2d(f, N);
```

The exact solution in this example is

\f[
    u(x,y)=\sin(\pi x)\sin(\pi y).
\f]

## Finite-Difference Reference

```cpp
num::Matrix u_fd = num::pde::poisson2d_fd(f, N);
```

Use `poisson2d_fd` as a direct finite-difference reference for small grids.

## Size Constraint

The DST implementation uses FFTs of length \f$2(N+1)\f$. The current built-in
radix-2 path requires \f$N+1\f$ to be a power of two, for example
\f$N=7,15,31,63,127\f$.
