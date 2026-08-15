# 2D Heat Equation {#page_heat_demo}

**Source:** `examples/heat_demo.cpp`

Solves \f$\partial_t u = \kappa\,\nabla^2 u\f$ on \f$[0,1]^2\f$ with zero Dirichlet BCs.
A Gaussian blob set by `num::gaussian2d` + `num::fill_grid` diffuses under the standard
5-point Laplacian (`num::pde::diffusion_step_2d_dirichlet`).
Three snapshots are rendered as a side-by-side heatmap with `num::plt::heatmap`.

---

## Code

```cpp
#include "numerics.hpp"
#include <string>

int main() {
    const int    N     = 64;
    const double kappa = 1.0, sigma = 0.06, T_end = 0.012;
    const double h     = 1.0 / (N + 1);
    const double dt    = 0.20 * h * h;          // < 0.25 h² stability limit
    const int    nstep = static_cast<int>(T_end / dt) + 1;
    const double coeff = kappa * dt / (h * h);

    num::Vector u(static_cast<std::size_t>(N) * N);
    num::fill_grid(u, N, h, [=](double x, double y) {
        return num::gaussian2d(x, y, 0.5, 0.5, sigma);
    });

    num::Vector u0 = u, u_mid;
    for (int s = 0; s < nstep; ++s) {
        if (s == nstep / 2) { u_mid = u; }
        num::pde::diffusion_step_2d_dirichlet(u, N, coeff);
    }

    num::plt::subplot(3, 1);
    num::plt::heatmap(u0,    N, h); num::plt::title("t = 0");
    num::plt::xlabel("x");  num::plt::ylabel("y");  num::plt::next();

    num::plt::heatmap(u_mid, N, h);
    num::plt::title("t = " + std::to_string((nstep/4)*dt).substr(0,6));
    num::plt::xlabel("x");  num::plt::next();

    num::plt::heatmap(u,     N, h);
    num::plt::title("t = " + std::to_string(T_end).substr(0,6));
    num::plt::xlabel("x");

    num::plt::savefig("heat_demo.png");
}
```

---

## Figure

\image html heat_demo.png "Column: t=0 (sharp Gaussian) → t=T/4 (spreading) → t=T (diffused)." width=500px

---

## Library features used

| Feature | Role |
|---------|------|
| `num::pde::diffusion_step_2d_dirichlet` | Explicit Euler step: u += coeff · Lap(u), Dirichlet BCs |
| `num::gaussian2d` | Isotropic Gaussian IC: \f$\exp(-r^2/2\sigma^2)\f$ |
| `num::fill_grid` | Populates an NxN grid from f(x, y) |
| `num::plt::heatmap` | Renders a 2D field as a gnuplot `pm3d map` panel |
| `num::plt::subplot` | Arranges multiple panels side by side |
