FetchContent_MakeAvailable(googletest)

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
)

target_link_libraries(numerics_tests PRIVATE
    numerics
    GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(numerics_tests)
