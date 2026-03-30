# report/CMakeLists.txt
# Builds gen_report and defines the `report` custom target.
#
# Prerequisites: NUMERICS_BUILD_TESTS and NUMERICS_BUILD_BENCHMARKS must be ON.
# Usage: cmake --build build --target report

if(NOT NUMERICS_BUILD_TESTS OR NOT NUMERICS_BUILD_BENCHMARKS)
    message(STATUS "Report: skipped (needs NUMERICS_BUILD_TESTS and NUMERICS_BUILD_BENCHMARKS)")
    return()
endif()

# ── gen_report executable ──────────────────────────────────────────────────────
add_executable(gen_report gen_report.cpp)
target_compile_features(gen_report PRIVATE cxx_std_17)

# ── Build-info fragment (written at cmake configure time) ─────────────────────
set(REPORT_OUT "${CMAKE_SOURCE_DIR}/output")

string(TIMESTAMP BUILD_DATE "%Y-%m-%d %H:%M UTC")

foreach(_var BLAS_FOUND FFTW3_FOUND OpenMP_CXX_FOUND NUMERICS_HAS_CUDA NUMERICS_HAS_MPI)
    if(${_var})
        set(${_var}_STR "true")
    else()
        set(${_var}_STR "false")
    endif()
endforeach()
set(REPORT_HAS_BLAS   ${BLAS_FOUND_STR})
set(REPORT_HAS_FFTW   ${FFTW3_FOUND_STR})
set(REPORT_HAS_OMP    ${OpenMP_CXX_FOUND_STR})
set(REPORT_HAS_CUDA   ${NUMERICS_HAS_CUDA_STR})
set(REPORT_HAS_MPI    ${NUMERICS_HAS_MPI_STR})

if(NUMERICS_HAS_LAPACK)
    set(REPORT_HAS_LAPACK "true")
else()
    set(REPORT_HAS_LAPACK "false")
endif()

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
endif()

# ── System information ─────────────────────────────────────────────────────────
cmake_host_system_information(RESULT REPORT_SYSTEM_NAME QUERY OS_NAME)
cmake_host_system_information(RESULT REPORT_CPU         QUERY PROCESSOR_DESCRIPTION)
cmake_host_system_information(RESULT REPORT_RAM_MB      QUERY TOTAL_PHYSICAL_MEMORY)

if(NUMERICS_HAS_CUDA)
    execute_process(
        COMMAND nvidia-smi --query-gpu=name --format=csv,noheader
        OUTPUT_VARIABLE _GPU_NAME
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    if(_GPU_NAME)
        set(REPORT_GPU "${_GPU_NAME}")
    else()
        set(REPORT_GPU "CUDA (name unavailable)")
    endif()
else()
    set(REPORT_GPU "n/a")
endif()

file(MAKE_DIRECTORY ${REPORT_OUT})
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/build_info.json.in
    ${REPORT_OUT}/build_info.json
    @ONLY
)

# ── `report` target ────────────────────────────────────────────────────────────
set(TEST_JSON   ${REPORT_OUT}/test_results.json)
set(BENCH_JSON  ${REPORT_OUT}/bench_results.json)
set(PLOTS_DIR   ${REPORT_OUT}/plots)
set(REPORT_MD   ${REPORT_OUT}/REPORT.md)
set(REPORT_HTML ${REPORT_OUT}/REPORT.html)

add_custom_target(report
    COMMENT "==> Generating ${REPORT_MD}"

    # 1. Create output directories
    COMMAND ${CMAKE_COMMAND} -E make_directory ${PLOTS_DIR}

    # 2. Run tests -- cmake -P wrapper so a test failure doesn't abort the target
    COMMAND ${CMAKE_COMMAND} -E echo "-- Running tests..."
    COMMAND ${CMAKE_COMMAND}
            -DTEST_BIN=$<TARGET_FILE:numerics_tests>
            -DOUT_JSON=${TEST_JSON}
            -P ${CMAKE_CURRENT_SOURCE_DIR}/run_tests.cmake

    # 3. Run benchmarks (JSON + PNG plots)
    COMMAND ${CMAKE_COMMAND} -E echo "-- Running benchmarks (this may take a few minutes)..."
    COMMAND $<TARGET_FILE:numerics_bench>
            --benchmark_out=${BENCH_JSON}
            --benchmark_out_format=json
            --benchmark_min_time=0.1s
            --report=${PLOTS_DIR}

    # 4. Assemble report
    COMMAND ${CMAKE_COMMAND} -E echo "-- Assembling report..."
    COMMAND $<TARGET_FILE:gen_report>
            ${CMAKE_CURRENT_SOURCE_DIR}/template.md
            ${REPORT_MD}
            ${TEST_JSON}
            ${BENCH_JSON}
            ${REPORT_OUT}/build_info.json
            ${PLOTS_DIR}

    COMMAND ${CMAKE_COMMAND} -E echo "==> Report ready: ${REPORT_MD}"

    # 5. Sync report into docs/
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_SOURCE_DIR}/docs/report
    COMMAND ${CMAKE_COMMAND} -E copy ${REPORT_MD} ${CMAKE_SOURCE_DIR}/docs/report/REPORT.md
    COMMAND ${CMAKE_COMMAND} -E copy ${REPORT_OUT}/REPORT.html ${CMAKE_SOURCE_DIR}/docs/report/index.html
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${PLOTS_DIR} ${CMAKE_SOURCE_DIR}/docs/report/plots
    COMMAND ${CMAKE_COMMAND} -E echo "==> Docs updated: docs/report/"

    DEPENDS gen_report numerics_tests numerics_bench
    VERBATIM
)
