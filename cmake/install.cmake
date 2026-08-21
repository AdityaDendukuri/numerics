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
install(TARGETS numerics_raw_kernel numerics_kernel numerics_solvers numerics_ode numerics_pde numerics_spectral numerics numerics_io
    EXPORT      NumericsTargets
    ARCHIVE     DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT full
    LIBRARY     DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT full
    RUNTIME     DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT full
)


# Header-only Raw Kernel headers component
install(FILES
    include/kernel/raw.hpp
    include/core/types.hpp
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/kernel
    COMPONENT kernel-raw
)

# Kernel module headers component
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
