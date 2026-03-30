FetchContent_MakeAvailable(raylib)

add_executable(quantum_demo
    main.cpp
    src/quantum/statevector.cpp
    src/quantum/circuit.cpp
)
target_include_directories(quantum_demo PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include
)
target_link_libraries(quantum_demo PRIVATE numerics raylib)
target_compile_features(quantum_demo PRIVATE cxx_std_17)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(quantum_demo PRIVATE -g -O0)
else()
    target_compile_options(quantum_demo PRIVATE -O3 -march=native)
endif()

add_custom_target(run_quantum_demo
    COMMAND quantum_demo
    DEPENDS quantum_demo
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)
