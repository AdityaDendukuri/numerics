FetchContent_MakeAvailable(benchmark)

# Google Benchmark 1.8.3 uses __COUNTER__ in a preprocessor expression.  New
# Clang versions diagnose that as a C2y extension, while the dependency enables
# -pedantic-errors unconditionally.  Keep the workaround on the third-party
# target rather than weakening diagnostics for numerics.
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    foreach(_benchmark_target benchmark benchmark_main)
        if(TARGET ${_benchmark_target})
            target_compile_options(${_benchmark_target} PRIVATE -Wno-c2y-extensions)
        endif()
    endforeach()
endif()

find_program(GNUPLOT_EXECUTABLE gnuplot)

add_executable(numerics_bench
    main.cpp
    bench_memory.cpp
    bench_linalg.cpp
    bench_banded.cpp
    bench_autovec.cpp
    bench_fft.cpp
    bench_factorization.cpp
    bench_eigen.cpp
    bench_svd.cpp
    bench_kernel.cpp
)

add_executable(bench_convergence bench_convergence.cpp)
target_link_libraries(bench_convergence PRIVATE numerics)

target_link_libraries(numerics_bench PRIVATE
    numerics
    benchmark::benchmark
)

if(GNUPLOT_EXECUTABLE)
    target_compile_definitions(numerics_bench PRIVATE NUMERICS_HAS_GNUPLOT)
    message(STATUS "gnuplot: ${GNUPLOT_EXECUTABLE} — --plot flag enabled")
else()
    message(STATUS "gnuplot: not found — --plot flag will be unavailable")
endif()

# bench_autovec.cpp is compiled without the auto-vectoriser to produce a true
# scalar baseline. -mavx2 stays active (inherited from the numerics target) but
# the tree-vectoriser and SLP vectoriser are disabled.
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set_source_files_properties(bench_autovec.cpp PROPERTIES
        COMPILE_FLAGS "-fno-tree-vectorize -fno-tree-slp-vectorize"
    )
endif()
