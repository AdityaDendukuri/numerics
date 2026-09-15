# SIMD — compile-time backend selection
include(CheckCXXCompilerFlag)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|i686")
    check_cxx_compiler_flag("-mavx2 -mfma" COMPILER_SUPPORTS_AVX2)
    if(COMPILER_SUPPORTS_AVX2)
        target_compile_options(numerics_backend_simd INTERFACE -mavx2 -mfma)
        target_compile_definitions(numerics_backend_simd INTERFACE NUMERICS_HAS_AVX2 NUMERICS_HAS_SIMD)
        message(STATUS "SIMD:  AVX-256 + FMA")
    else()
        message(STATUS "SIMD:  none (compiler lacks -mavx2)")
    endif()
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64|ARM64|AARCH64")
    target_compile_definitions(numerics_backend_simd INTERFACE NUMERICS_HAS_NEON NUMERICS_HAS_SIMD)
    message(STATUS "SIMD:  ARM NEON")
else()
    message(STATUS "SIMD:  none (unknown arch: ${CMAKE_SYSTEM_PROCESSOR})")
endif()

# Fused multiply-add. clang contracts `a*b + c` within a statement by default;
# GCC turns contraction off under strict `-std=c++NN`, which halves the ceiling
# of every kernel that accumulates products. Attached here rather than to the
# kernel target so numerics::kernel stays free of compile options.
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(numerics_backend_simd INTERFACE -ffp-contract=fast)
endif()

# Host cache sizes for kernel::gemm's blocking (NUM_K_L1_BYTES, NUM_K_L2_BYTES).
# The kernel carries conservative defaults; these only sharpen them, and are
# skipped when cross-compiling since the build host says nothing about the
# target.
if(NOT CMAKE_CROSSCOMPILING)
    set(_l1 "")
    set(_l2 "")
    if(APPLE)
        execute_process(COMMAND sysctl -n hw.perflevel0.l1dcachesize
                        OUTPUT_VARIABLE _l1 OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
        if(NOT _l1)
            execute_process(COMMAND sysctl -n hw.l1dcachesize
                            OUTPUT_VARIABLE _l1 OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
        endif()
        execute_process(COMMAND sysctl -n hw.perflevel0.l2cachesize
                        OUTPUT_VARIABLE _l2 OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
        if(NOT _l2)
            execute_process(COMMAND sysctl -n hw.l2cachesize
                            OUTPUT_VARIABLE _l2 OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
        endif()
    elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        execute_process(COMMAND getconf LEVEL1_DCACHE_SIZE
                        OUTPUT_VARIABLE _l1 OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
        execute_process(COMMAND getconf LEVEL2_CACHE_SIZE
                        OUTPUT_VARIABLE _l2 OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
    endif()
    if(_l1 MATCHES "^[0-9]+$" AND _l1 GREATER 0)
        target_compile_definitions(numerics_backend_simd INTERFACE NUM_K_L1_BYTES=${_l1})
    endif()
    if(_l2 MATCHES "^[0-9]+$" AND _l2 GREATER 0)
        target_compile_definitions(numerics_backend_simd INTERFACE NUM_K_L2_BYTES=${_l2})
    endif()
    if(_l1 OR _l2)
        message(STATUS "Cache: L1d ${_l1} B, L2 ${_l2} B (kernel::gemm blocking)")
    endif()
endif()
