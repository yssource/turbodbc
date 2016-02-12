include_directories(${CMAKE_CURRENT_SOURCE_DIR})

file(GLOB_RECURSE TEST_FILES "*.cpp")

add_executable(cpp_odbc_test ${TEST_FILES})

add_dependencies(cpp_odbc_test
    refresh_cmake_configuration
)

target_link_libraries(cpp_odbc_test
	cpp_odbc
    cppunit_toolbox
    cppunit_runner_gmock
    cppunit
    ${Odbc_LIBRARIES}
)

add_test(Library cpp_odbc_test --xml-output ${CMAKE_BINARY_DIR}/cpp_odbc_test.xml)