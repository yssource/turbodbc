find_package(PythonInterp REQUIRED)
add_test(
    NAME PyTest
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMAND ${PYTHON_EXECUTABLE} -m unittest discover
)
set_tests_properties(
    PyTest
    PROPERTIES
    ENVIRONMENT PYTHONPATH=${CMAKE_BINARY_DIR}
)