# examples/CMakeLists.txt
# Umbrella target to build all examples: cmake --build build --target numerics_examples
add_custom_target(numerics_examples ALL)

# Helper function to register an example target with full control
function(add_numerics_example TARGET_NAME SOURCE_FILE)
    add_executable(${TARGET_NAME} ${SOURCE_FILE} ${ARGN})
    target_link_libraries(${TARGET_NAME} PRIVATE numerics::numerics)
    add_dependencies(numerics_examples ${TARGET_NAME})
endfunction()

# ==============================================================================
# Explicitly Registered Showcase Examples
# Add your custom example target here!
# ==============================================================================

add_numerics_example(00_core_storage_and_helpers    00_core_storage_and_helpers.cpp)
add_numerics_example(01_direct_factorizations        01_direct_factorizations.cpp)
add_numerics_example(02_iterative_krylov_solvers      02_iterative_krylov_solvers.cpp)
add_numerics_example(03_resolvent_and_expv           03_resolvent_and_expv.cpp)
add_numerics_example(04_eigen_and_svd                04_eigen_and_svd.cpp)
add_numerics_example(05_symplectic_nbody_ode         05_symplectic_nbody_ode.cpp)
add_numerics_example(06_pde_poisson_solver           06_pde_poisson_solver.cpp)
add_numerics_example(07_spectral_fft_transforms     07_spectral_fft_transforms.cpp)
add_numerics_example(08_root_finding_and_quadrature  08_root_finding_and_quadrature.cpp)
add_numerics_example(09_mcmc_bayesian_sampling       09_mcmc_bayesian_sampling.cpp)
add_numerics_example(10_banded_and_spd_operators     10_banded_and_spd_operators.cpp)
add_numerics_example(11_terminal_ascii_plot         11_terminal_ascii_plot.cpp)
add_numerics_example(12_hessenberg_resolvent_benchmark 12_hessenberg_resolvent_benchmark.cpp)
add_numerics_example(13_talbot_spectral_validation   13_talbot_spectral_validation.cpp)
