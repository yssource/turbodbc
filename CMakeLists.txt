project(pydbc CXX)
cmake_minimum_required(VERSION 2.8.7)

include(default_build_setup)

find_package(UsePsapp REQUIRED)
exclude_from_coverage(psapp)

include_directories(SYSTEM ${Boost_INCLUDE_DIRS})

find_package(odbc)
include_directories(SYSTEM ${Odbc_INCLUDE_DIR})

find_package(cpp_odbc)

include_directories(Library)
add_subdirectory(Library)

if(BUILD_TESTING)
    enable_testing()

    find_package(cppunit_toolbox)
    add_subdirectory(Test)
endif()