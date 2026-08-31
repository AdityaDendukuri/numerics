# FindFFTW3.cmake -- find FFTW3 and create FFTW3::FFTW3 imported target.
#
# Sets:
#   FFTW3_FOUND
#   FFTW3_LIBRARY
#   FFTW3_INCLUDE_DIR
#
# Defines target:
#   FFTW3::FFTW3

if(TARGET FFTW3::FFTW3)
    return()
endif()

# 1. Try pkg-config first
find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
    pkg_check_modules(_FFTW3 QUIET fftw3)
    if(_FFTW3_FOUND)
        find_library(FFTW3_LIBRARY NAMES fftw3
            HINTS ${_FFTW3_LIBRARY_DIRS})
        find_path(FFTW3_INCLUDE_DIR fftw3.h
            HINTS ${_FFTW3_INCLUDE_DIRS})
    endif()
endif()

# 2. Manual search fallback
if(NOT FFTW3_LIBRARY)
    find_library(FFTW3_LIBRARY NAMES fftw3
        HINTS
            /opt/homebrew/lib
            /usr/local/lib
            /usr/lib
            /usr/lib/x86_64-linux-gnu
    )
endif()
if(NOT FFTW3_INCLUDE_DIR)
    find_path(FFTW3_INCLUDE_DIR fftw3.h
        HINTS
            /opt/homebrew/include
            /usr/local/include
            /usr/include
    )
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(FFTW3
    REQUIRED_VARS FFTW3_LIBRARY FFTW3_INCLUDE_DIR)

if(FFTW3_FOUND AND NOT TARGET FFTW3::FFTW3)
    add_library(FFTW3::FFTW3 UNKNOWN IMPORTED)
    set_target_properties(FFTW3::FFTW3 PROPERTIES
        IMPORTED_LOCATION             "${FFTW3_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${FFTW3_INCLUDE_DIR}"
    )
endif()

mark_as_advanced(FFTW3_LIBRARY FFTW3_INCLUDE_DIR)
