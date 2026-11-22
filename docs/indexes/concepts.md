# All concepts

A concept such as `linear_operator` holds when a type provides the right operations. A law such as SPD cannot be checked by the compiler, so it is declared with `assume_spd` or `make_spd`.

## Scalars
field scalar_function differentiable_function

## Spaces
vector_space inner_product_space linear_subspace_of

## Operators
linear_operator self_adjoint_operator psd_operator spd_operator sparse_convertible

## Matrices
matrix_space

## Laws
claims with_law law::self_adjoint law::psd law::spd law::diagonally_dominant

## Storage layout
contiguous dense_row_major csr banded tridiagonal

## Discrete structures
equivalence_relation incidence_structure weighted_incidence addressable_priority_queue

## Index spaces
square_extent_2d cartesian_index_space_2d periodic_neighbourhood_2d

## Fields and grids
scalar_field_like solvable_field

## Ordinary differential equations
vec_field

## Partial differential equations
assemblable_grid_operator

## Spatial acceleration
position_accessor_2d neighbor_query_2d smoothing_kernel

## Stochastic
categorical_sampling energy_difference

## Quadrature
contour_rule

## Spectral
transform_plan

## Statistics
streaming_accumulator moment_accumulator

## Root finding
bracketable_function

