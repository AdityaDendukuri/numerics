# Linear algebra

Factorizations, solvers, eigenvalues and the SVD. Every factorization is solved by `solve(F, b, x)`, and the raw-pointer overloads work on caller-owned buffers.

## The solve protocol <linear/solve.hpp>
num::factorization num::solve_transpose

## LU <linear/factorization/lu.hpp>
num::lu num::lu_result num::no_pivot num::no_pivot_structure num::lower num::upper num::det num::inverse num::seq::lu num::lapack::lu num::lu_factor num::lu_factor_blocked num::lu_solve num::lu_invert num::lu_no_pivot num::lu_no_pivot_solve_multiple num::lu_no_pivot_solve_transpose_multiple

## Mixed precision <linear/factorization/mixed_lu.hpp>
num::mixed_lu num::mixed_precision num::mixed_precision_structure

## Cholesky <linear/factorization/cholesky.hpp>
num::cholesky num::cholesky_result num::cholesky_update num::cholesky_downdate num::lapack::cholesky num::unsafe::cholesky num::linear::is_spd num::linear::make_spd num::cholesky_blocked num::cholesky_block num::cholesky_solve

## QR <linear/factorization/qr.hpp>
num::qr num::qr_result num::seq::qr num::lapack::qr num::householder_vector num::householder_vector_strided num::householder_left num::householder_right num::qr_factor_blocked num::qr_form_block num::qr_apply_block_left num::qr_workspace num::qr_block

## Hessenberg <linear/factorization/hessenberg.hpp>
num::hessenberg num::hessenberg_decomposition num::hessenberg_project num::hessenberg_back_project num::seq::hessenberg num::lapack::hessenberg num::hessenberg_shifted_factor num::hessenberg_shifted_substitute num::hessenberg_shifted_solve

## Banded <linear/banded/banded.hpp>
num::band_mat num::banded_lu_result num::banded_matvec num::banded_gemv num::banded_norm1 num::banded_factor num::banded_solve num::solve

## Tridiagonal <linear/factorization/thomas.hpp>
num::thomas num::seq::thomas num::lapack::thomas

## Complex tridiagonal <linear/factorization/tridiag_complex.hpp>
num::complex_tri_diag

## Block tridiagonal <linear/factorization/block_tridiagonal.hpp>
num::block_lu_factor num::block_cholesky_factor num::factor_block_lu num::factor_block_cholesky num::refactor_block_lu_suffix num::refactor_block_cholesky_suffix num::solve_in_place num::solve_transpose_in_place

## Choosing a factorization <linear/factorization/factor.hpp>
num::sparse num::sparse_structure num::blocks num::block_structure num::similar_factor

## Factor reuse <linear/factorization/reuse.hpp>
num::corrected_lu num::suffix_block_lu

## Low-rank updates <linear/factorization/woodbury.hpp>
num::woodbury_solver num::low_rank_difference num::low_rank_update num::update_inverse_rows num::inverse_rows_workspace

## Inverse diagonal <linear/factorization/inverse_diagonal.hpp>
num::inverse_diagonal num::inverse_diagonal_workspace num::inverse_principal_block num::selected_inverse num::safe_add

## Probed inverse diagonal <linear/factorization/probed_inverse_diagonal.hpp>
num::inverse_diagonal_options

## Conditioning <linear/condition.hpp>
num::rcond num::inverse_norm1_estimate num::opnorm1

## Matrix properties <linear/matrix_properties.hpp>
num::linear::is_symmetric num::linear::make_symmetric num::linear::symmetry_error num::linear::relative_symmetry_error

## Sparse checks <linear/debug.hpp>
num::linear::debug::verify_sparse_structure

## Conjugate gradient <linear/solvers/cg.hpp>
num::cg num::cg_options num::unsafe::cg num::krylov_result

## Preconditioned CG <linear/solvers/pcg.hpp>
num::pcg num::pcg_options

## MINRES <linear/solvers/minres.hpp>
num::minres num::minres_options

## GMRES <linear/solvers/gmres.hpp>
num::gmres num::gmres_options

## Stationary iterations <linear/solvers/jacobi.hpp>
num::jacobi

## Gauss-Seidel <linear/solvers/gauss_seidel.hpp>
num::gauss_seidel

## Solver results <linear/solvers/solver_result.hpp>
num::solver_result

## Solver callables <linear/solvers/linear_solver.hpp>
num::linear_solver

## Diagonal preconditioners <linear/solvers/preconditioner.hpp>
num::jacobi_preconditioner num::make_jacobi_preconditioner

## ILU(0) <linear/solvers/ilu.hpp>
num::ilu0_preconditioner num::make_ilu0_preconditioner num::ilu0_factor num::csr_lu_solve num::csr_diagonal_positions

## Chebyshev <linear/solvers/chebyshev.hpp>
num::chebyshev_preconditioner num::chebyshev_preconditioner_from_below num::make_chebyshev_preconditioner num::estimate_largest_eigenvalue

## Automatic solver <linear/solvers/auto_linear.hpp>
num::auto_linear_solver num::auto_linear_options

## Hessenberg resolvent <linear/solvers/hessenberg_resolvent.hpp>
num::hessenberg_resolvent num::hessenberg_shifted_lu num::shift num::solve_batch

## Sparse resolvent <linear/solvers/sparse_resolvent.hpp>
num::sparse_resolvent num::sparse_shifted_lu num::sparse_resolvent_options num::sparse_resolvent_available

## Automatic resolvent <linear/solvers/auto_resolvent.hpp>
num::auto_resolvent num::auto_shifted_lu num::auto_resolvent_options

## Matrix exponential <linear/expv/expv.hpp>
num::expv

## KLU <linear/sparse/klu.hpp>
num::klu_factorization num::klu_available

## UMFPACK <linear/sparse/umfpack.hpp>
num::umfpack_factor num::umfpack_available

## Sparse operator <linear/sparse/sparse_op.hpp>
num::operators::sparse_op

## Symmetric eigenvalues <linear/eigen/jacobi_eig.hpp>
num::eig_sym num::eigen_result num::seq::eig_sym num::omp::eig_sym num::lapack::eig_sym num::unsafe::eig_sym num::jacobi_rotation

## Lanczos <linear/eigen/lanczos.hpp>
num::lanczos num::lanczos_result num::unsafe::lanczos num::sqrt_lanczos num::inverse_sqrt_lanczos num::lanczos_action_result

## Power iteration <linear/eigen/power.hpp>
num::power_iteration num::inverse_iteration num::rayleigh_iteration num::power_result

## SVD <linear/svd/svd.hpp>
num::svd num::svd_truncated num::svd_result num::seq::svd num::lapack::svd

## Subspaces <linear/subspace.hpp>
num::mgs_columns num::dispatch::subspace::mgs_orthogonalize num::dispatch::subspace::arnoldi_step

## Matrices from graphs <linear/graph/laplacian.hpp>
num::linear::laplacian num::linear::dense_laplacian num::linear::markov_generator num::linear::dense_markov_generator num::linear::to_dense_adjacency num::linear::to_sparse_adjacency num::linear::to_multigraph

## Graph levels <linear/graph/levels.hpp>
num::graph_distance_levels

## Approximate Cholesky <linear/graph/randommat/approxchol.hpp>
num::randommat::factorize num::randommat::exact num::randommat::ac1 num::randommat::ac2 num::randommat::act num::randommat::gao_kyng_spielman_2023::ac num::randommat::gao_kyng_spielman_2023::ac2

## Approximate Cholesky preconditioner <linear/graph/randommat/preconditioner.hpp>
num::approx_chol_preconditioner num::grounded_approx_chol_factor num::randommat::approx_chol_preconditioner num::randommat::approxchol_preconditioner num::randommat::grounded_approx_chol_factor num::randommat::grounded_approxchol_factor num::randommat::to_approxchol_graph

## Approximate Cholesky solves <linear/graph/randommat/solve.hpp>
num::randommat::solve

## Approximate Cholesky types <linear/graph/randommat/types.hpp>
num::randommat::cholesky_factor num::randommat::factor_column num::randommat::factor_entry num::randommat::get_entry num::randommat::graph_edge num::randommat::neighbor num::randommat::adjacency_list
