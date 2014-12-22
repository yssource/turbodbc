cmake_minimum_required(VERSION 2.8.7)
project(pydbc_intern CXX)
set(pydbc_VERSION 0.0.1-dev)

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

    find_package(PythonInterp REQUIRED)
    add_test(
        NAME PyTest
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        COMMAND ${PYTHON_EXECUTABLE} "${CMAKE_SOURCE_DIR}/PyTest/test_suite.py"
    )
endif()

file(COPY pydbc DESTINATION ${CMAKE_BINARY_DIR})

add_subdirectory(Wheel)