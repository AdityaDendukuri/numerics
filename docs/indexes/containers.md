# All containers

Grouped by what each does. Types own their storage and are over-aligned; the free functions write into caller-provided destinations.

## Scalar types <core/types.hpp>
num::real num::idx num::cplx num::array num::static_array num::view num::unordered_map num::map num::unordered_set num::set num::append num::scalar_fn num::vector_fn num::to_idx

## Machine precision and comparing numbers <core/types.hpp>
num::eps num::unit_roundoff num::inf num::nan num::tiny num::huge num::max num::min num::close num::ulp_distance num::basic_tolerance num::tolerance

## Dense vectors <container/vector.hpp>
num::vec num::vec2_view num::copy_to

## Dense matrices <container/matrix.hpp>
num::mat

## Vector arithmetic <container/vector_ops.hpp>
num::scale num::axpy num::axpby num::axpbyz num::add num::dot num::norm

## Reductions <container/reduce.hpp>
num::sum num::l1_norm num::linf_norm

## Matrix arithmetic <container/matrix_ops.hpp>
num::matmul num::matvec num::matadd num::gemm num::gemv

## Rank-1 updates and triangular systems <container/dense.hpp>
num::ger num::trsv_lower num::trsv_upper

## Constructing matrices <linear/matrix_utils.hpp>
num::identity num::eye num::zeros num::ones num::unit_vector num::identity_columns num::diagonal_matrix num::transpose

## Diagonals <linear/matrix_utils.hpp>
num::diagonal num::set_diagonal

## Scaling and summing <linear/matrix_utils.hpp>
num::scale_elements num::scale_rows num::divide_elements num::divide_rows num::accu

## Gather and scatter <linear/matrix_utils.hpp>
num::gather num::scatter

## Sparse matrices <linear/sparse/sparse.hpp>
num::spmat num::spmat::from_triplets num::spmat::from_csc num::sparse_matvec num::dense num::spmat::nnz num::diagonal_similarity num::sparse_diagonal_similarity num::sparse_congruence num::symmetric_part num::with_explicit_diagonal num::characteristic_matrix num::sparse_linear_combination

## Bilinear forms with sparse matrices <linear/sparse/bilinear.hpp>
num::sparse_bilinear num::sparse_bilinear_trace

## Small matrices of fixed size <container/small_matrix.hpp>
num::small_vec num::small_matrix num::givens_rotation

## Lattice points <container/lattice_point.hpp>
num::lattice_point

## Aligned memory <container/util/aligned_storage.hpp>
num::aligned_array num::make_aligned num::make_aligned_for_overwrite num::is_storage_aligned num::storage_alignment num::assume_storage_aligned

## Orthogonal polynomials <container/util/math.hpp>
num::legendre num::assoc_legendre num::laguerre num::assoc_laguerre num::hermite

## Special functions <container/util/math.hpp>
num::bessel_j num::bessel_y num::bessel_i num::sph_bessel_j num::sph_bessel_y num::beta num::gaussian2d

## Ranges and sampling <container/util/math.hpp>
num::linspace num::logspace num::int_range num::rng_state num::rng_uniform num::rng_normal num::rng_int

## Constants <container/util/math.hpp>
num::pi num::two_pi num::half_pi num::inv_pi num::e num::ln2 num::sqrt2 num::sqrt3 num::phi

## Integer powers <container/util/integer_pow.hpp>
num::ipow
