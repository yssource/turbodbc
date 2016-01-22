include_directories(${CMAKE_CURRENT_SOURCE_DIR})

file(GLOB_RECURSE TEST_FILES "*.cpp")

add_executable(turbodbc_test ${TEST_FILES})

add_dependencies(turbodbc_test
    refresh_cmake_configuration
)

target_link_libraries(turbodbc_test
	turbodbc_intern
	cpp_odbc
    cppunit_toolbox
    cppunit_runner_gmock
    cppunit
    ${Odbc_LIBRARIES}
)

add_test(Library turbodbc_test)
