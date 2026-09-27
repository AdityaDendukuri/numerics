# The documentation site: tools/refdoc.py writes docs/reference from the header comments,
# then mkdocs builds the site. Needs `pip install libclang mkdocs-material`.
find_package(Python3 REQUIRED COMPONENTS Interpreter)

set(_refdoc_environment NUMERICS_BUILD=${CMAKE_BINARY_DIR})
if(NUMERICS_BUILD_IO)
    list(APPEND _refdoc_environment
        NUMERICS_DOC_INCLUDE=${nlohmann_json_SOURCE_DIR}/include)
endif()

add_custom_target(docs
    COMMAND ${CMAKE_COMMAND} -E env ${_refdoc_environment}
            ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tools/refdoc.py build
    COMMAND ${Python3_EXECUTABLE} -m mkdocs build --strict
            --config-file ${CMAKE_SOURCE_DIR}/mkdocs.yml
            --site-dir ${CMAKE_CURRENT_BINARY_DIR}/site
    COMMAND ${CMAKE_COMMAND} -E touch ${CMAKE_CURRENT_BINARY_DIR}/site/.nojekyll
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Generating the reference and building the documentation site"
    VERBATIM)
