cmake_minimum_required(VERSION 2.8.7)
project(turbodbc_intern CXX)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR})

include(default_build_setup)

find_package(Boost REQUIRED COMPONENTS python system)
include_directories(SYSTEM ${Boost_INCLUDE_DIRS})

find_package(PythonLibs REQUIRED)
include_directories(SYSTEM ${PYTHON_INCLUDE_DIRS})

find_package(odbc REQUIRED)
include_directories(SYSTEM ${Odbc_INCLUDE_DIR})

find_package(cpp_odbc REQUIRED)

include_directories(Library)
add_subdirectory(Library)

if(BUILD_TESTING)
    enable_testing()

    find_package(cppunit_toolbox)
    add_subdirectory(Test)

    add_subdirectory(PyTest PyTest)
endif()

file(COPY turbodbc DESTINATION ${CMAKE_BINARY_DIR})

add_subdirectory(Wheel)