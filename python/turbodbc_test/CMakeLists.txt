find_package(PythonInterp REQUIRED)
add_test(
    NAME turbodbc_integration_test
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMAND ${PYTHON_EXECUTABLE} -m unittest discover
)
set_tests_properties(
    turbodbc_integration_test
    PROPERTIES
    ENVIRONMENT PYTHONPATH=${CMAKE_BINARY_DIR}
)