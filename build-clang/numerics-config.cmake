
####### Expanded from @PACKAGE_INIT@ by configure_package_config_file() #######
####### Any changes to this file will be overwritten by the next CMake run ####
####### The input file was NumericsConfig.cmake.in                            ########

get_filename_component(PACKAGE_PREFIX_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../" ABSOLUTE)

macro(set_and_check _var _file)
  set(${_var} "${_file}")
  if(NOT EXISTS "${_file}")
    message(FATAL_ERROR "File or directory ${_file} referenced by variable ${_var} does not exist !")
  endif()
endmacro()

macro(check_required_components _NAME)
  foreach(comp ${${_NAME}_FIND_COMPONENTS})
    if(NOT ${_NAME}_${comp}_FOUND)
      if(${_NAME}_FIND_REQUIRED_${comp})
        set(${_NAME}_FOUND FALSE)
      endif()
    endif()
  endforeach()
endmacro()

####################################################################################

include(CMakeFindDependencyMacro)

# Re-find optional dependencies that were compiled into the library.
# Each block mirrors the detection logic in cmake/*.cmake so that the
# imported targets referenced in NumericsTargets.cmake are recreated.

if(TRUE)
    find_dependency(BLAS)
endif()

if(ON)
    find_dependency(LAPACK)
endif()

if(TRUE)
    find_dependency(OpenMP)
endif()

if(TRUE)
    find_dependency(MPI)
endif()

if(ON)
    find_dependency(PkgConfig)
    pkg_check_modules(KLU REQUIRED IMPORTED_TARGET klu)
endif()

if(ON)
    find_dependency(PkgConfig)
    pkg_check_modules(UMFPACK REQUIRED IMPORTED_TARGET umfpack)
endif()

if(TRUE)
    # FindFFTW3.cmake is installed alongside this file.
    list(PREPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}")
    find_dependency(FFTW3)
endif()

include("${CMAKE_CURRENT_LIST_DIR}/NumericsTargets.cmake")
check_required_components(numerics)
