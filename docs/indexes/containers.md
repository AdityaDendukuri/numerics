# All containers

Grouped by what each does. Types own their storage and are over-aligned; the free functions write into caller-provided destinations.

## Scalars and vocabulary <core/types.hpp>
num::real num::idx num::cplx num::array num::static_array num::view num::unordered_map num::map num::unordered_set num::set num::append num::scalar_fn num::vector_fn num::to_idx

## Dense vectors <container/vector.hpp>
num::basic_vec num::vec num::cvec num::vec2_view num::copy_to

## Dense matrices <container/matrix.hpp>
num::basic_mat num::mat

## Vector arithmetic <container/vector_ops.hpp>
num::scale num::axpy num::axpby num::axpbyz num::add num::dot num::norm

## Reductions <container/reduce.hpp>
num::sum num::l1_norm num::linf_norm

## Matrix arithmetic <container/matrix_ops.hpp>
num::matmul num::matvec num::matadd

## Rank-1 and triangular solves <container/dense.hpp>
num::ger num::trsv_lower num::trsv_upper

## Matrix construction <linear/matrix_utils.hpp>
num::identity num::eye num::zeros num::ones num::unit_vector num::identity_columns num::diagonal_matrix num::transpose

## Diagonals <linear/matrix_utils.hpp>
num::diagonal num::set_diagonal

## Scaling and accumulation <linear/matrix_utils.hpp>
num::scale_elements num::scale_rows num::divide_elements num::divide_rows num::accu

## Gather and scatter <linear/matrix_utils.hpp>
num::gather num::scatter

## Sparse matrices <linear/sparse/sparse.hpp>
num::spmat num::spmat::from_triplets num::spmat::from_csc num::sparse_matvec num::dense num::spmat::nnz num::diagonal_similarity num::sparse_diagonal_similarity num::sparse_congruence num::symmetric_part

## Small fixed-size <container/small_matrix.hpp>
num::small_vec num::small_matrix num::givens_rotation

## Discrete indices <container/multi_index.hpp>
num::multi_index

## Over-aligned storage <container/util/aligned_storage.hpp>
num::aligned_array num::make_aligned num::make_aligned_for_overwrite num::is_storage_aligned num::storage_alignment num::assume_storage_aligned

## Orthogonal polynomials <container/util/math.hpp>
num::legendre num::assoc_legendre num::laguerre num::assoc_laguerre num::hermite

## Special functions <container/util/math.hpp>
num::bessel_j num::bessel_y num::bessel_i num::sph_bessel_j num::sph_bessel_y num::beta num::gaussian2d

## Ranges and sampling <container/util/math.hpp>
num::linspace num::logspace num::int_range num::rng_state num::rng_uniform num::rng_normal num::rng_int

## Constants <container/util/math.hpp>
num::pi num::two_pi num::half_pi num::inv_pi num::e num::ln2 num::sqrt2 num::sqrt3 num::phi

## Misc numerics <container/util/integer_pow.hpp>
num::ipow
