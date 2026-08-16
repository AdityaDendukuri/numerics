# examples/CMakeLists.txt
# Auto-discovers all *.cpp example files and creates executable targets linked to numerics::numerics.
# Also creates a custom meta-target 'numerics_examples' to build all examples at once.

file(GLOB EXAMPLE_SOURCES CONFIGURE_DEPENDS "*.cpp")

add_custom_target(numerics_examples ALL)

foreach(SRC ${EXAMPLE_SOURCES})
    get_filename_component(TARGET_NAME ${SRC} NAME_WE)
    add_executable(${TARGET_NAME} ${SRC})
    target_link_libraries(${TARGET_NAME} PRIVATE numerics::numerics)
    add_dependencies(numerics_examples ${TARGET_NAME})
endforeach()
