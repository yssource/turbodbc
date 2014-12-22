find_package(PythonInterp REQUIRED)
add_test(
    NAME PyTest
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMAND ${PYTHON_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/test_suite.py"
)
set_tests_properties(
    PyTest
    PROPERTIES
    ENVIRONMENT PYTHONPATH=${CMAKE_BINARY_DIR}
)