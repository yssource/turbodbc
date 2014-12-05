project(cpp_odbc CXX)
cmake_minimum_required(VERSION 2.8.7)

include(default_build_setup)

#find_package(Boost REQUIRED COMPONENTS system locale)
find_package(UsePsapp REQUIRED)
include_directories(SYSTEM ${Boost_INCLUDE_DIRS})

find_package(odbc)
include_directories(SYSTEM ${Odbc_INCLUDE_DIR})

include_directories(Library)
add_subdirectory(Library)

if(BUILD_TESTING)
    enable_testing()

    find_package(cppunit_toolbox)
    add_subdirectory(Test)
endif()