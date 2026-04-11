# install.cmake -- install rules for numerics-core.
#
# Installs the numerics library, public headers, and CMake package files so
# that downstream projects can use:
#
#   find_package(numerics REQUIRED)
#   target_link_libraries(my_target PRIVATE numerics::numerics)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

# Library artifact + export set
install(TARGETS numerics
    EXPORT      NumericsTargets
    ARCHIVE     DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY     DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME     DESTINATION ${CMAKE_INSTALL_BINDIR}
)

# Public headers (preserves directory structure under include/)
install(DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# CMake package files
configure_package_config_file(
    cmake/NumericsConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/NumericsConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/numerics
)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/NumericsConfigVersion.cmake
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
    ${CMAKE_CURRENT_BINARY_DIR}/NumericsConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/NumericsConfigVersion.cmake
    cmake/FindFFTW3.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/numerics
)
