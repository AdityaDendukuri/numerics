# FFTW3 -- spectral transforms backend
#
# Searches for a system FFTW3 installation via pkg-config then find_library.
# Falls back gracefully (fftw -> seq) if absent.
#
# Sets: NUMERICS_HAS_FFTW compile definition on the numerics target.
#
# Install:
#   macOS:  brew install fftw
#   Ubuntu: apt install libfftw3-dev

if(NOT NUMERICS_USE_FFTW)
    return()
endif()

# 1. Try pkg-config with an imported target (handles lib path + includes automatically)
find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
    pkg_check_modules(FFTW3 QUIET IMPORTED_TARGET fftw3)
endif()

# 2. Manual fallback: find_library + find_path
if(NOT FFTW3_FOUND)
    find_library(FFTW3_LIBRARY NAMES fftw3
        HINTS /usr/local/lib /opt/homebrew/lib /usr/lib /usr/lib/x86_64-linux-gnu
    )
    find_path(FFTW3_INCLUDE_DIR fftw3.h
        HINTS /usr/local/include /opt/homebrew/include /usr/include
    )
    if(FFTW3_LIBRARY AND FFTW3_INCLUDE_DIR)
        set(FFTW3_FOUND TRUE)
        # Create a minimal imported target to unify both paths below
        add_library(PkgConfig::FFTW3 UNKNOWN IMPORTED)
        set_target_properties(PkgConfig::FFTW3 PROPERTIES
            IMPORTED_LOCATION             "${FFTW3_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${FFTW3_INCLUDE_DIR}"
        )
    endif()
endif()

if(NOT FFTW3_FOUND)
    message(STATUS "FFTW3: not found -- Backend::fftw falls back to Backend::seq")
    message(STATUS "       Install: brew install fftw  |  apt install libfftw3-dev")
    return()
endif()

target_link_libraries(numerics PUBLIC PkgConfig::FFTW3)
target_compile_definitions(numerics PUBLIC NUMERICS_HAS_FFTW)
message(STATUS "FFTW3: found  (${FFTW3_LIBRARY}${FFTW3_LIBRARIES})")
