# report/run_tests.cmake
# Runs the test binary and writes gtest JSON output.
# Called via: cmake -P run_tests.cmake
#             with -DTEST_BIN=... -DOUT_JSON=...
#
# The report is generated regardless of pass/fail: a non-zero exit code
# from the test binary is logged but does not abort the report target.

execute_process(
    COMMAND "${TEST_BIN}" "--gtest_output=json:${OUT_JSON}"
    RESULT_VARIABLE result
)
if(result EQUAL 0)
    message(STATUS "Tests: all passed")
else()
    message(STATUS "Tests: ${result} failure(s) -- see ${OUT_JSON} for details")
endif()
