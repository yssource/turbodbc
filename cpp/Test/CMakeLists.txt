include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(SYSTEM $ENV{CPPUNIT_TOOLBOX_INCLUDE_DIR})

file(GLOB_RECURSE TEST_FILES "*.cpp")

link_directories($ENV{GOOGLETEST_LIB_DIR})

add_executable(turbodbc_test ${TEST_FILES})

add_dependencies(turbodbc_test
    refresh_cmake_configuration
)

target_link_libraries(turbodbc_test
	turbodbc_intern
	cpp_odbc
    cppunit_toolbox
    cppunit
    gmock_main
    ${Odbc_LIBRARIES}
)

add_test(Library turbodbc_test)
