# examples/CMakeLists.txt
# Umbrella target to build all examples: cmake --build build --target numerics_examples
add_custom_target(numerics_examples ALL)

# Helper function to register an example target with full control
function(add_numerics_example TARGET_NAME SOURCE_FILE)
    add_executable(${TARGET_NAME} ${SOURCE_FILE} ${ARGN})
    target_link_libraries(${TARGET_NAME} PRIVATE numerics::numerics)
    add_dependencies(numerics_examples ${TARGET_NAME})
endfunction()

# The recorded examples save their results as JSON through examples/io/, built on
# nlohmann/json. Only the examples use it; the numerics library does not.
find_package(nlohmann_json 3.11 CONFIG QUIET)
if(NOT nlohmann_json_FOUND)
    include(FetchContent)
    FetchContent_Declare(nlohmann_json
        URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
    FetchContent_MakeAvailable(nlohmann_json)
endif()

function(add_numerics_io_example TARGET_NAME)
    add_executable(${TARGET_NAME} ${TARGET_NAME}.cpp)
    target_include_directories(${TARGET_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(${TARGET_NAME} PRIVATE numerics::numerics nlohmann_json::nlohmann_json)
    add_dependencies(numerics_examples ${TARGET_NAME})
endfunction()

add_numerics_io_example(00_vectors_and_matrices)
add_numerics_io_example(01_linear_systems)
add_numerics_io_example(02_iterative_methods)
add_numerics_io_example(03_matrix_exponential)
add_numerics_io_example(04_eigenvalues)
add_numerics_io_example(05_differential_equations)
add_numerics_io_example(06_poisson_equation)
add_numerics_io_example(07_fourier_transforms)
add_numerics_io_example(08_roots_and_integrals)
add_numerics_io_example(09_statistics)
add_numerics_io_example(10_banded_systems)
add_numerics_example(11_terminal_plot 11_terminal_plot.cpp)
add_numerics_io_example(12_many_shifted_systems)
add_numerics_io_example(13_inverse_laplace_transform)
add_numerics_io_example(14_positive_definite_matrices)
add_numerics_io_example(15_diffusion_equation)
add_numerics_io_example(16_updating_a_factorization)
add_numerics_io_example(17_low_rank_updates)
add_numerics_io_example(18_condition_numbers)
add_numerics_io_example(19_diagonal_of_the_inverse)
add_numerics_io_example(20_blas_and_kernels)
add_numerics_io_example(21_graphs)
add_numerics_io_example(22_heat_equation)
add_numerics_io_example(23_frequency_response)
