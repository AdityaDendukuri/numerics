FetchContent_MakeAvailable(googletest)

# Configuration-time architecture checks: optional discovery must never mutate
# the dependency-free foundational targets.
get_target_property(_kernel_links numerics_kernel INTERFACE_LINK_LIBRARIES)
get_target_property(_kernel_definitions numerics_kernel INTERFACE_COMPILE_DEFINITIONS)
get_target_property(_kernel_options numerics_kernel INTERFACE_COMPILE_OPTIONS)
if(_kernel_links OR _kernel_definitions OR _kernel_options)
    message(FATAL_ERROR "numerics::kernel acquired a transitive dependency or backend setting")
endif()

get_target_property(_core_links numerics_core INTERFACE_LINK_LIBRARIES)
get_target_property(_core_definitions numerics_core INTERFACE_COMPILE_DEFINITIONS)
get_target_property(_core_options numerics_core INTERFACE_COMPILE_OPTIONS)
if(NOT _core_links STREQUAL "numerics_kernel" OR _core_definitions OR _core_options)
    message(FATAL_ERROR "numerics::core must depend only on numerics::kernel")
endif()

add_executable(numerics_tests
    test_vector.cpp
    test_matrix.cpp
    test_solvers.cpp
    test_banded.cpp
    test_factorization.cpp
    test_analysis.cpp
    test_fft.cpp
    test_ode.cpp
    test_eigen.cpp
    test_svd.cpp
    test_fields.cpp
    test_solve.cpp
    test_umbrella.cpp
    test_debug.cpp
    test_graph_ds.cpp
    test_math_spine.cpp
    test_randommat.cpp
)



target_link_libraries(numerics_tests PRIVATE
    numerics
    GTest::gtest_main
)
if(TARGET numerics_io)
    target_link_libraries(numerics_tests PRIVATE numerics_io)
    target_compile_definitions(numerics_tests PRIVATE NUMERICS_HAS_JSON=1)
endif()

include(GoogleTest)
gtest_discover_tests(numerics_tests)

# The tier-0 kernel must stay usable on its own: compiled against a copy of
# include/kernel alone, and linked against nothing. Anything the kernel starts
# depending on -- a container, the algebra, a backend symbol -- breaks this build.
set(KERNEL_ONLY_INCLUDE "${CMAKE_CURRENT_BINARY_DIR}/kernel_only_include")
file(REMOVE_RECURSE "${KERNEL_ONLY_INCLUDE}")
file(COPY "${CMAKE_SOURCE_DIR}/include/kernel" DESTINATION "${KERNEL_ONLY_INCLUDE}")

add_executable(numerics_kernel_standalone test_kernel_standalone.cpp)
target_include_directories(numerics_kernel_standalone PRIVATE "${KERNEL_ONLY_INCLUDE}")
target_compile_features(numerics_kernel_standalone PRIVATE cxx_std_20)
add_test(NAME Kernel.Standalone COMMAND numerics_kernel_standalone)

# The mathematical protocol is independently header-only and dependency-free.
# Compile against a copy of include/core alone so no container or backend can
# become an accidental transitive requirement.
set(MATH_CORE_ONLY_INCLUDE "${CMAKE_CURRENT_BINARY_DIR}/math_core_only_include")
file(REMOVE_RECURSE "${MATH_CORE_ONLY_INCLUDE}")
file(COPY "${CMAKE_SOURCE_DIR}/include/core" DESTINATION "${MATH_CORE_ONLY_INCLUDE}")

add_executable(numerics_math_core_standalone test_math_core_standalone.cpp)
target_include_directories(numerics_math_core_standalone PRIVATE "${MATH_CORE_ONLY_INCLUDE}")
target_compile_features(numerics_math_core_standalone PRIVATE cxx_std_20)
add_test(NAME MathCore.Standalone COMMAND numerics_math_core_standalone)

add_test(NAME Package.CoreDependencyFree
    COMMAND ${CMAKE_COMMAND}
        -DNUMERICS_BUILD_DIR=${CMAKE_BINARY_DIR}
        -DNUMERICS_SOURCE_DIR=${CMAKE_SOURCE_DIR}
        -DNUMERICS_CXX_COMPILER=${CMAKE_CXX_COMPILER}
        -DNUMERICS_PACKAGE_HAS_MPI=${NUMERICS_HAS_MPI}
        -DNUMERICS_PACKAGE_HAS_CUDA=${NUMERICS_HAS_CUDA}
        -P ${CMAKE_CURRENT_SOURCE_DIR}/test_core_package.cmake
)
