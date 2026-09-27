# All kernels

num::kernel: raw loops over pointers and callables, templated on the scalar, allocating nothing.

## Vector construction <kernel/vector.hpp>
num::kernel::copy num::kernel::fill num::kernel::copy_strided num::kernel::scale_copy_strided num::kernel::swap num::kernel::swap_strided

## Vector arithmetic <kernel/vector.hpp>
num::kernel::scale num::kernel::axpy num::kernel::axpy_strided num::kernel::axpby num::kernel::axpbyz num::kernel::add num::kernel::hadamard_mul num::kernel::hadamard_div num::kernel::inv num::kernel::clamp

## Reductions <kernel/vector.hpp>
num::kernel::dot num::kernel::sum num::kernel::norm num::kernel::norm_sq num::kernel::norm_sq_strided num::kernel::l1_norm num::kernel::linf_norm num::kernel::argmax_abs

## Fused reductions <kernel/vector.hpp>
num::kernel::axpy_norm_sq num::kernel::dot2 num::kernel::dot_norm_sq num::kernel::linear_combination_norm_sq num::kernel::dot2_result num::kernel::dot_norm_result

## Matrix-vector products <kernel/dense.hpp>
num::kernel::matvec num::kernel::matvec_transpose num::kernel::gbmv num::kernel::ger

## Matrix-matrix products <kernel/dense.hpp>
num::kernel::gemm num::kernel::gemm_config num::kernel::gemm_workspace num::kernel::gemm_transpose_left num::kernel::syrk_lower num::kernel::transpose

## Triangular solves <kernel/dense.hpp>
num::kernel::trsv_lower num::kernel::trsv_upper num::kernel::trsv_lower_inplace num::kernel::trsv_upper_inplace num::kernel::trsv_transpose_lower num::kernel::trsv_transpose_upper num::kernel::trsm_lower_inplace num::kernel::trsm_unit_lower_inplace num::kernel::trsm_lower_transpose_inplace num::kernel::trsm_unit_lower_transpose_inplace num::kernel::trsm_upper_inplace num::kernel::trsm_upper_transpose_inplace num::kernel::trsm_lower_transpose_right_inplace

## Orthogonalization <kernel/dense.hpp>
num::kernel::mgs_columns num::kernel::project_columns num::kernel::column_dot num::kernel::combine_columns num::kernel::rotate_columns num::kernel::swap_rows

## LU without pivoting <kernel/dense.hpp>
num::kernel::lu_no_pivot num::kernel::lu_no_pivot_solve_multiple num::kernel::lu_no_pivot_solve_transpose_multiple

## Cholesky <kernel/factor.hpp>
num::kernel::cholesky num::kernel::cholesky_blocked num::kernel::cholesky_solve num::kernel::cholesky_batched num::kernel::cholesky_solve_batched num::kernel::cholesky_invert

## LU with partial pivoting <kernel/factor.hpp>
num::kernel::lu_factor num::kernel::lu_factor_blocked num::kernel::lu_solve num::kernel::lu_invert

## Banded <kernel/factor.hpp>
num::kernel::banded_factor num::kernel::banded_solve

## Givens and Jacobi rotations <kernel/rotations.hpp>
num::kernel::rotg num::kernel::rot num::kernel::jacobi_rotation

## Householder and QR <kernel/rotations.hpp>
num::kernel::householder_vector num::kernel::householder_vector_strided num::kernel::householder_left num::kernel::householder_right num::kernel::qr_form_block num::kernel::qr_apply_block_left num::kernel::qr_factor_blocked num::kernel::qr_workspace num::kernel::qr_block

## Sparse products <kernel/sparse.hpp>
num::kernel::spmv num::kernel::spmv_axpy num::kernel::spmm

## Incomplete factorization <kernel/sparse.hpp>
num::kernel::csr_diagonal_positions num::kernel::ilu0_factor num::kernel::csr_lu_solve

## Krylov <kernel/krylov.hpp>
num::kernel::cg num::kernel::pcg num::kernel::krylov_result

## Real and complex <kernel/complex.hpp>
num::kernel::matvec_real_complex num::kernel::matvec_transpose_into_complex num::kernel::hessenberg_shifted_factor num::kernel::hessenberg_shifted_substitute num::kernel::hessenberg_shifted_solve
