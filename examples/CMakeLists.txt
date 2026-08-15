cmake_minimum_required(VERSION 3.20)
project(numerics_examples LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Debug and Diagnostic Options for Examples
option(EXAMPLES_DEBUG_MODE "Build examples with Debug symbols (-g, -O0)" ON)
option(EXAMPLES_ENABLE_SANITIZERS "Enable Address and UndefinedBehavior Sanitizers (-fsanitize=address,undefined)" OFF)

# Link against root CMake target or installed package
if(TARGET numerics)
    set(NUMERICS_TARGET numerics)
elseif(TARGET numerics::numerics)
    set(NUMERICS_TARGET numerics::numerics)
else()
    find_package(numerics REQUIRED)
    set(NUMERICS_TARGET numerics::numerics)
endif()

# ------------------------------------------------------------------------------
# 1-Line Helper Function: add_example(target_name)
# Usage: add_example(my_experiment) -> compiles examples/my_experiment.cpp
# ------------------------------------------------------------------------------
function(add_example NAME)
    set(SRC_FILE "${CMAKE_CURRENT_SOURCE_DIR}/${NAME}.cpp")
    if(NOT EXISTS "${SRC_FILE}")
        set(SRC_FILE "${CMAKE_CURRENT_SOURCE_DIR}/${NAME}/main.cpp")
    endif()

    if(NOT EXISTS "${SRC_FILE}")
        message(FATAL_ERROR "add_example: Source file for '${NAME}' not found at ${SRC_FILE}")
    endif()

    add_executable(${NAME} "${SRC_FILE}")
    target_link_libraries(${NAME} PRIVATE ${NUMERICS_TARGET})

    # Apply debug/optimization compiler options
    if(EXAMPLES_DEBUG_MODE)
        target_compile_options(${NAME} PRIVATE -g -O0)
        target_compile_definitions(${NAME} PRIVATE NUMERICS_DEBUG)
    else()
        target_compile_options(${NAME} PRIVATE -O3)
        target_compile_definitions(${NAME} PRIVATE NDEBUG)
    endif()

    # Address & Undefined Behavior Sanitizer flag
    if(EXAMPLES_ENABLE_SANITIZERS)
        target_compile_options(${NAME} PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
        target_link_options(${NAME} PRIVATE -fsanitize=address,undefined)
    endif()
endfunction()

# ------------------------------------------------------------------------------
# Auto-Discovery: Automatically creates a target for every .cpp file in examples/!
# ------------------------------------------------------------------------------
file(GLOB EXAMPLE_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
foreach(SRC ${EXAMPLE_SOURCES})
    get_filename_component(EX_NAME ${SRC} NAME_WE)
    add_example(${EX_NAME})
endforeach()
