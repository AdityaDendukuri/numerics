# SPH Kernels {#page_sph_kernel}

`include/spatial/sph_kernel.hpp` provides `num::SPHKernel<Dim>` for 2D and 3D
SPH density, viscosity, heat-diffusion, and pressure-gradient terms.

---

## Kernel Set

The dimension-dependent constants are specialized for `Dim=2` and `Dim=3`.
The formulas are otherwise shared.

---

## Routine Reference

```cpp
template<int Dim>   // Dim = 2 or 3
struct num::SPHKernel {
    static float W(float r, float h);
    static float dW_dr(float r, float h);
    static float Spiky_dW_dr(float r, float h);
    static std::array<float, Dim> Spiky_gradW(std::array<float, Dim> r_vec, float r, float h);
};
```

All functions are `static` -- no state, no instantiation needed.

---

## Kernels

### Cubic Spline -- density

Support = \f$2h\f$, \f$q = r/h\f$.

\f[
W(r,h) = \sigma \begin{cases}
  1 - \tfrac{3}{2}q^2 + \tfrac{3}{4}q^3 & q \le 1 \\
  \tfrac{1}{4}(2-q)^3                    & 1 < q \le 2 \\
  0                                       & q > 2
\end{cases}
\f]

\f[
\sigma_{\text{2D}} = \frac{10}{7\pi h^2}, \qquad
\sigma_{\text{3D}} = \frac{1}{\pi h^3}
\f]

`dW_dr` returns the radial derivative \f$\partial W/\partial r \le 0\f$, used in the
**Morris et al. 1997 SPH Laplacian**:

\f[
\nabla^2 A_i \approx \sum_j \frac{m_j}{\rho_j}
  \frac{2(A_i - A_j)\,r_{ij}\,(\partial W/\partial r)}{r_{ij}^2 + \varepsilon^2}
\f]

(viscosity and heat diffusion both use this formula).

### Spiky Kernel -- pressure gradient

Maintains \f$\nabla W \ne 0\f$ as \f$r \to 0\f$ to prevent particle clustering.

\f[
\frac{\partial W_{\text{spiky}}}{\partial r} = \begin{cases}
-\dfrac{15}{16\pi h^5}(2h-r)^2 & \text{2D} \\[6pt]
-\dfrac{45}{\pi (2h)^6}(2h-r)^2 & \text{3D}
\end{cases}
\f]

`Spiky_gradW` returns the gradient vector:

\f[
\nabla W = \frac{\partial W/\partial r}{r}\,\mathbf{r}
\f]

using `std::array<float, Dim>` for both input `r_vec` and output.

---

## Dimension specialization

The only dimension-dependent parts are isolated in `detail::CubicSigma<Dim>` and
`detail::SpikyDW<Dim>` in the anonymous `num::detail` namespace.
Both are fully specialized for `Dim=2` and `Dim=3`.

---

## Example

```cpp
// 2D density update
const float W0 = num::SPHKernel<2>::W(0.0f, h);
const float w  = m * num::SPHKernel<2>::W(r, h);

// 2D pressure gradient
auto g = num::SPHKernel<2>::Spiky_gradW({rx, ry}, r, h);
float gx = g[0], gy = g[1];

// 3D pressure gradient
auto g3 = num::SPHKernel<3>::Spiky_gradW({rx, ry, rz}, r, h);

// Morris Laplacian (viscosity / heat, same formula for 2D and 3D)
float lap = 2.0f * num::SPHKernel<Dim>::Spiky_dW_dr(r, h) * r / (r2 + eps2);
```

## SPH Terms

| Kernel | Term |
|---|---|
| `SPHKernel<Dim>::W` | Density interpolation |
| `SPHKernel<Dim>::dW_dr` | Morris viscosity and heat Laplacian |
| `SPHKernel<Dim>::Spiky_dW_dr` | Spiky pressure derivative |
| `SPHKernel<Dim>::Spiky_gradW` | Pressure-gradient vector |
