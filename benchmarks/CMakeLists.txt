FetchContent_MakeAvailable(benchmark)

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
)

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
