# install.cmake -- install rules for numerics-core.
#
# Installs the numerics library, public headers, and CMake package files so
# that downstream projects can use:
#
#   find_package(numerics REQUIRED)
#   target_link_libraries(my_target PRIVATE numerics::numerics)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

# Library artifacts + export set
set(NUMERICS_INSTALL_TARGETS
    numerics_kernel numerics_core
    numerics_backend_blas numerics_backend_lapack numerics_backend_openmp
    numerics_backend_fftw numerics_backend_suitesparse numerics_backend_simd
    numerics_backends numerics
    numerics_mpi
    numerics_solvers numerics_ode numerics_pde numerics_spectral numerics_plot
)
if(TARGET numerics_cuda)
    list(APPEND NUMERICS_INSTALL_TARGETS numerics_cuda)
endif()
if(TARGET numerics_io)
    list(APPEND NUMERICS_INSTALL_TARGETS numerics_io)
endif()
install(TARGETS ${NUMERICS_INSTALL_TARGETS}
    EXPORT      NumericsTargets
    ARCHIVE     DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT full
    LIBRARY     DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT full
    RUNTIME     DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT full
)


# Kernel module headers component. This is the raw, bare-metal tier in full:
# vector/dense/sparse/rotations/factor/krylov all live directly under
# num::kernel with no further split, so one directory install covers it.
install(DIRECTORY include/kernel/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/kernel
    COMPONENT kernel
)

# All public headers (preserves directory structure under include/)
install(DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    COMPONENT full
)


# CMake package files
configure_package_config_file(
    cmake/NumericsConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/numerics-config.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/numerics
)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/numerics-config-version.cmake
    VERSION       ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

# Export targets file (NumericsTargets.cmake)
install(EXPORT NumericsTargets
    FILE        NumericsTargets.cmake
    NAMESPACE   numerics::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/numerics
)

# Config, version, and FindFFTW3 module
install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/numerics-config.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/numerics-config-version.cmake
    cmake/FindFFTW3.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/numerics
)
