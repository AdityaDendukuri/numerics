# All kernels

num::kernel: the BLAS layer, raw loops over pointers, templated on the scalar, allocating nothing.

## Vector construction <kernel/vector.hpp>
num::kernel::copy num::kernel::fill num::kernel::copy_strided num::kernel::scale_copy_strided num::kernel::swap num::kernel::swap_strided

## Givens rotations <kernel/vector.hpp>
num::kernel::rotg num::kernel::rot

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
num::kernel::trsv_lower num::kernel::trsv_upper num::kernel::trsv_lower_inplace num::kernel::trsv_upper_inplace num::kernel::trsv_transpose_lower num::kernel::trsm_lower_inplace num::kernel::trsm_unit_lower_inplace num::kernel::trsm_lower_transpose_inplace num::kernel::trsm_unit_lower_transpose_inplace num::kernel::trsm_upper_inplace num::kernel::trsm_upper_transpose_inplace num::kernel::trsm_lower_transpose_right_inplace

## Block products and row swaps <kernel/dense.hpp>
num::kernel::project_columns num::kernel::combine_columns num::kernel::swap_rows

## Sparse products <kernel/sparse.hpp>
num::kernel::spmv num::kernel::spmv_axpy num::kernel::spmm

## Real and complex <kernel/complex.hpp>
num::kernel::matvec_real_complex num::kernel::matvec_transpose_into_complex
