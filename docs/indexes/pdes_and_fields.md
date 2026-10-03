# PDEs and fields

Grids, fields, stencils and the solvers built on them. The concepts are on the [Concepts](concepts.md) page.

## 2D grids <fields/grid2d.hpp>
num::grid2d

## 2D fields <fields/scalar_field_2d.hpp>
num::scalar_field_2d

## 3D grids <fields/grid3d.hpp>
num::grid_3d

## 3D fields <fields/field3d.hpp>
num::scalar_field_3d num::vector_field_3d

## Stencils <pde/stencil.hpp>
num::laplacian_stencil_2d num::laplacian_stencil_2d_4th num::laplacian_stencil_2d_periodic num::neg_laplacian_3d num::gradient_3d num::divergence_3d num::curl_3d num::fill_grid num::row_fiber_sweep num::col_fiber_sweep num::sample_2d_periodic

## Grid operators <pde/grid_operators.hpp>
num::operators::laplacian_2d num::operators::backward_euler_2d num::operators::to_sparse_matrix

## Diffusion <pde/diffusion.hpp>
num::pde::diffusion_step_2d num::pde::diffusion_step_2d_dirichlet num::pde::diffusion_step_2d_4th_dirichlet num::pde::laplacian_sparse_2d num::pde::matrix_free_laplacian_2d num::pde::backward_euler_matrix num::pde::backward_euler_operator num::pde::backward_euler_operator_2d num::pde::matrix_free_backward_euler_2d num::pde::make_cg_solver

## ADI <pde/adi.hpp>
num::crank_nicolson_adi

## Poisson <pde/poisson.hpp>
num::pde::poisson2d num::pde::poisson2d_fd

## Field solvers <pde/field_solver.hpp>
num::field_solver num::magnetic_solver
