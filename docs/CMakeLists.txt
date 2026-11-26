# The documentation site: tools/refdoc.py writes docs/reference from the header comments,
# then mkdocs builds the site. The Python packages they need are pinned in
# docs/requirements.txt and installed into a private virtual environment in the build
# directory on the first docs build, so only a python3 interpreter has to be present.
find_package(Python3 REQUIRED COMPONENTS Interpreter)

set(_docs_venv ${CMAKE_CURRENT_BINARY_DIR}/venv)
if(WIN32)
    set(_docs_python ${_docs_venv}/Scripts/python.exe)
else()
    set(_docs_python ${_docs_venv}/bin/python)
endif()
set(_docs_requirements ${CMAKE_CURRENT_SOURCE_DIR}/requirements.txt)
set(_docs_env_stamp ${_docs_venv}/requirements.stamp)

# Re-runs only when requirements.txt changes.
add_custom_command(
    OUTPUT ${_docs_env_stamp}
    COMMAND ${Python3_EXECUTABLE} -m venv ${_docs_venv}
    COMMAND ${_docs_python} -m pip install --quiet --disable-pip-version-check
            -r ${_docs_requirements}
    COMMAND ${CMAKE_COMMAND} -E touch ${_docs_env_stamp}
    DEPENDS ${_docs_requirements}
    COMMENT "Setting up the documentation Python environment"
    VERBATIM)

set(_refdoc_environment NUMERICS_BUILD=${CMAKE_BINARY_DIR})

add_custom_target(docs
    COMMAND ${CMAKE_COMMAND} -E env ${_refdoc_environment}
            ${_docs_python} ${CMAKE_SOURCE_DIR}/tools/refdoc.py build
    COMMAND ${_docs_python} -m mkdocs build --strict
            --config-file ${CMAKE_SOURCE_DIR}/mkdocs.yml
            --site-dir ${CMAKE_CURRENT_BINARY_DIR}/site
    COMMAND ${CMAKE_COMMAND} -E touch ${CMAKE_CURRENT_BINARY_DIR}/site/.nojekyll
    DEPENDS ${_docs_env_stamp}
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Generating the reference and building the documentation site"
    VERBATIM)

# Reports doc comments that disagree with their declarations and public names missing from
# docs/indexes; fails on any problem. Names without a comment are listed but do not fail.
add_custom_target(docs-check
    COMMAND ${CMAKE_COMMAND} -E env ${_refdoc_environment}
            ${_docs_python} ${CMAKE_SOURCE_DIR}/tools/refdoc.py check
    DEPENDS ${_docs_env_stamp}
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Checking the doc comments and indexes"
    VERBATIM)
