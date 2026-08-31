set(_prefix "${NUMERICS_BUILD_DIR}/tests/core_package_install")
set(_consumer_build "${NUMERICS_BUILD_DIR}/tests/core_package_consumer_build")

file(REMOVE_RECURSE "${_prefix}" "${_consumer_build}")

set(_package_targets numerics)
if(NUMERICS_PACKAGE_HAS_MPI)
    list(APPEND _package_targets numerics_mpi)
endif()
if(NUMERICS_PACKAGE_HAS_CUDA)
    list(APPEND _package_targets numerics_cuda)
endif()

foreach(_target IN LISTS _package_targets)
    execute_process(
        COMMAND "${CMAKE_COMMAND}" --build "${NUMERICS_BUILD_DIR}" --target "${_target}"
        RESULT_VARIABLE _prerequisite_result
        OUTPUT_VARIABLE _prerequisite_output
        ERROR_VARIABLE _prerequisite_error
    )
    if(NOT _prerequisite_result EQUAL 0)
        message(FATAL_ERROR
            "package prerequisite ${_target} build failed:\n"
            "${_prerequisite_output}\n${_prerequisite_error}")
    endif()
endforeach()

execute_process(
    COMMAND "${CMAKE_COMMAND}" --install "${NUMERICS_BUILD_DIR}" --prefix "${_prefix}"
    RESULT_VARIABLE _install_result
    OUTPUT_VARIABLE _install_output
    ERROR_VARIABLE _install_error
)
if(NOT _install_result EQUAL 0)
    message(FATAL_ERROR "core package install failed:\n${_install_output}\n${_install_error}")
endif()

execute_process(
    COMMAND "${CMAKE_COMMAND}"
        -S "${NUMERICS_SOURCE_DIR}/tests/package_core_consumer"
        -B "${_consumer_build}"
        -DCMAKE_CXX_COMPILER=${NUMERICS_CXX_COMPILER}
        -DCMAKE_PREFIX_PATH=${_prefix}
        -DCMAKE_DISABLE_FIND_PACKAGE_BLAS=TRUE
        -DCMAKE_DISABLE_FIND_PACKAGE_LAPACK=TRUE
        -DCMAKE_DISABLE_FIND_PACKAGE_OpenMP=TRUE
        -DCMAKE_DISABLE_FIND_PACKAGE_MPI=TRUE
        -DCMAKE_DISABLE_FIND_PACKAGE_FFTW3=TRUE
        -DCMAKE_DISABLE_FIND_PACKAGE_PkgConfig=TRUE
        -DCMAKE_DISABLE_FIND_PACKAGE_CUDAToolkit=TRUE
    RESULT_VARIABLE _configure_result
    OUTPUT_VARIABLE _configure_output
    ERROR_VARIABLE _configure_error
)
if(NOT _configure_result EQUAL 0)
    message(FATAL_ERROR
        "dependency-free core consumer configuration failed:\n"
        "${_configure_output}\n${_configure_error}")
endif()

execute_process(
    COMMAND "${CMAKE_COMMAND}" --build "${_consumer_build}"
    RESULT_VARIABLE _build_result
    OUTPUT_VARIABLE _build_output
    ERROR_VARIABLE _build_error
)
if(NOT _build_result EQUAL 0)
    message(FATAL_ERROR "dependency-free core consumer build failed:\n${_build_output}\n${_build_error}")
endif()
