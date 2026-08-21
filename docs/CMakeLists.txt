find_package(Doxygen REQUIRED)

set(DOXYGEN_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
set(DOXYGEN_EXTRACT_ALL YES)
set(DOXYGEN_GENERATE_TREEVIEW YES)
set(DOXYGEN_DISABLE_INDEX NO)
set(DOXYGEN_FULL_SIDEBAR NO)
set(DOXYGEN_HTML_EXTRA_STYLESHEET
    "${CMAKE_CURRENT_SOURCE_DIR}/doxygen-awesome.css"
)
set(DOXYGEN_HTML_COLORSTYLE DARK)
set(DOXYGEN_IMAGE_PATH
    "${CMAKE_CURRENT_SOURCE_DIR}"
)
# Use MathJax to render \f[...\f] math without requiring a LaTeX installation
set(DOXYGEN_USE_MATHJAX YES)
set(DOXYGEN_MATHJAX_VERSION MathJax_3)
set(DOXYGEN_GENERATE_LATEX NO)
set(DOXYGEN_SEARCHENGINE YES)
set(DOXYGEN_GENERATE_TAGFILE "${CMAKE_CURRENT_BINARY_DIR}/numerics.tag")
set(DOXYGEN_EXAMPLE_PATH "${CMAKE_SOURCE_DIR}/examples")
set(DOXYGEN_EXAMPLE_RECURSIVE YES)
set(DOXYGEN_FILE_PATTERNS "*.hpp" "*.h" "*.md" "*.cpp")

# Source browser — generates HTML source pages and adds "Go to source" links
# on every class, function, and file documentation page.
set(DOXYGEN_SOURCE_BROWSER YES)
set(DOXYGEN_REFERENCED_BY_RELATION YES)  # "Referenced by" back-links on each symbol
set(DOXYGEN_REFERENCES_RELATION YES)     # "References" forward-links on each symbol
set(DOXYGEN_STRIP_CODE_COMMENTS NO)      # Keep comments visible in the source view

doxygen_add_docs(docs
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/examples
    ${CMAKE_CURRENT_SOURCE_DIR}/pages
    COMMENT "Generate API documentation"
)
