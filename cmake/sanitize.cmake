# Sanitizer support
# Usage: -DNUMERICS_SANITIZE=asan       (address + leak)
#        -DNUMERICS_SANITIZE=lsan       (leak only, faster)
#        -DNUMERICS_SANITIZE=ubsan      (undefined behaviour)
#        -DNUMERICS_SANITIZE=tsan       (thread races — useful for OpenMP/MPI work)
#        -DNUMERICS_SANITIZE=asan,ubsan (combine with commas)
#
# Notes:
#  - asan and tsan are mutually exclusive (different shadow-memory layouts)
#  - Incompatible with -DNUMERICS_ENABLE_CUDA=ON (CUDA runtime conflicts with ASan)
#  - Use a separate build dir: cmake -B build-san -DNUMERICS_SANITIZE=asan ...

if(NOT NUMERICS_SANITIZE)
    return()
endif()

if(NOT CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    message(WARNING "[sanitize] Sanitizers require GCC or Clang — skipping")
    return()
endif()

# Parse comma-separated list
string(REPLACE "," ";" _san_list "${NUMERICS_SANITIZE}")

set(_san_flags "")
set(_have_asan  OFF)
set(_have_tsan  OFF)

foreach(_san IN LISTS _san_list)
    string(TOLOWER "${_san}" _san)
    if(_san STREQUAL "asan")
        list(APPEND _san_flags -fsanitize=address -fno-omit-frame-pointer)
        set(_have_asan ON)
    elseif(_san STREQUAL "lsan")
        list(APPEND _san_flags -fsanitize=leak    -fno-omit-frame-pointer)
    elseif(_san STREQUAL "ubsan")
        list(APPEND _san_flags -fsanitize=undefined -fno-omit-frame-pointer)
    elseif(_san STREQUAL "tsan")
        list(APPEND _san_flags -fsanitize=thread  -fno-omit-frame-pointer)
        set(_have_tsan ON)
    else()
        message(WARNING "[sanitize] Unknown sanitizer '${_san}' — ignored")
    endif()
endforeach()

if(_have_asan AND _have_tsan)
    message(FATAL_ERROR "[sanitize] asan and tsan are mutually exclusive")
endif()

if(NUMERICS_HAS_CUDA AND (_have_asan OR _have_tsan))
    message(WARNING "[sanitize] ASan/TSan + CUDA may crash at runtime — proceed carefully")
endif()

if(_san_flags)
    # Deduplicate (e.g. duplicate -fno-omit-frame-pointer from multiple sanitizers)
    list(REMOVE_DUPLICATES _san_flags)

    add_compile_options(${_san_flags})
    add_link_options(${_san_flags})

    message(STATUS "[sanitize] Active: ${NUMERICS_SANITIZE}  flags: ${_san_flags}")
endif()
