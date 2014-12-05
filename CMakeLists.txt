project(cpp_odbc CXX)
cmake_minimum_required(VERSION 2.8.7)

include(default_build_setup)

find_package(Boost REQUIRED COMPONENTS system locale thread serialization regex date_time)
include_directories(SYSTEM ${Boost_INCLUDE_DIRS})

include_directories(Library)
add_subdirectory(Library)

if(BUILD_TESTING)
    enable_testing()

    find_package(cppunit_toolbox)
    add_subdirectory(Test)
endif()