# Spatial Acceleration {#page_spatial}

The `spatial` module provides cell lists, Verlet lists, periodic lattice indexing, and SPH smoothing kernels for neighbour queries in particle and lattice simulations.

Coordinates live in a scalar field, so every structure is templated on `num::field`. An integer coordinate type is rejected at compile time, since it would assign particles to the wrong cells and report nothing.

---

## 1. Cell List

Bins particles into a uniform grid of edge \f$h\f$, so a query touches only the nine cells around a point instead of all \f$N\f$ particles:

\f[
\mathcal{O}(N^2) \;\longrightarrow\; \mathcal{O}(N)
\f]

### Build from a position accessor

```cpp
num::cell_list_2d<double> cells(/*cell_size=*/1.0, /*xmin=*/0.0, /*xmax=*/10.0,
                                                /*ymin=*/0.0, /*ymax=*/10.0);

auto position = [&](int i) { return std::pair<double, double>{x[i], y[i]}; };

cells.build(position, n); // Counting sort over cell ids in O(N).
```

The list never owns coordinates. The accessor lets positions live in a struct of arrays, an array of structs, or a simulation's own particle type.

### Query candidates near a point

```cpp
cells.query(px, py, [&](int j) {
    const double dx = x[j] - px;
    const double dy = y[j] - py;
    if ((dx * dx) + (dy * dy) < cutoff * cutoff) {
        accumulate(j); // j is a candidate; the caller applies the exact test.
    }
});
```

`query` visits candidates rather than returning a container, so nothing allocates in the inner loop. Candidates outside the cutoff are still reported, and the caller applies the distance test.

---

## 2. Verlet List

Caches neighbour pairs within \f$r_c + s\f$ for a cutoff \f$r_c\f$ and skin \f$s\f$, so the list survives several steps before it must be rebuilt:

```cpp
num::verlet_list_2d<double> verlet(/*cutoff=*/2.5, /*skin=*/0.5);

verlet.build(position, n, cells); // Uses the cell list to find pairs.

for (int step = 0; step < steps; ++step) {
    integrate(dt);

    if (verlet.needs_rebuild(position, n)) { // True once a particle has moved
        cells.build(position, n);            // more than skin/2.
        verlet.build(position, n, cells);
    }

    for (int i = 0; i < n; ++i) {
        for (int j : verlet.neighbors(i)) { // Cached pairs, no search.
            apply_force(i, j);
        }
    }
}
```

The skin trades memory for rebuild frequency. A larger skin holds more pairs and rebuilds less often.

---

## 3. Periodic Lattice

Precomputes the four periodic neighbours of every site on an \f$N \times N\f$ lattice, so a sweep evaluates no modulus:

```cpp
num::pbc_lattice_2d lattice(/*N=*/64);

const int row = 3, col = 5;
const int i = (row * lattice.N) + col;

const int above = lattice.up[i]; // Row -1, wrapping at the boundary.
const int below = lattice.dn[i];
const int left  = lattice.lt[i];
const int right = lattice.rt[i];
```

Stepping up then down returns to the original site. `num::spatial::debug::verify_lattice_symmetry` checks that the tables are mutually inverse, which a table built with the wrong modulus is not.

---

## 4. SPH Smoothing Kernels

Cubic spline kernel \f$W(r, h)\f$ in two or three dimensions, normalized over its support:

\f[
\int_0^{2h} W(r,h) \, 2\pi r \, dr = 1, \qquad W(r,h) = 0 \ \text{ for } r > 2h
\f]

```cpp
const float w    = num::sph_kernel<2>::W(r, h);          // Kernel value.
const float dwdr = num::sph_kernel<2>::dW_dr(r, h);      // Radial derivative.
const float spiky = num::sph_kernel<2>::Spiky_dW_dr(r, h); // Pressure gradient variant.
```

Normalization is what makes an SPH sum reproduce a constant field. A kernel that is not normalized yields densities off by a constant factor, which looks like a physical result rather than an error:

```cpp
num::spatial::debug::verify_kernel_normalization<num::sph_kernel<2>>(h); // Integrates the kernel.
num::spatial::debug::verify_kernel_support<num::sph_kernel<2>>(h);       // Checks W(2h+, h) == 0.
```

---

## 5. Compile-Time Concepts

```cpp
auto position = [](int i) { return std::pair<double, double>{double(i), double(i)}; };

static_assert(num::position_accessor_2d<decltype(position), double>);
static_assert(num::neighbor_query_2d<num::cell_list_2d<double>, double>);
static_assert(num::smoothing_kernel<num::sph_kernel<2>, float>);
static_assert(num::periodic_lattice_2d<num::pbc_lattice_2d>);
```

---

## Complete Examples

- @ref page_pbc_lattice "Periodic boundary condition indexing"
- @ref page_sph_kernel "SPH kernel normalization"
- @ref page_connected_components "Spatial connected-component labeling"

