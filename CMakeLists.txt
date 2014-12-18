project(pydbc_intern CXX)
cmake_minimum_required(VERSION 2.8.7)

include(default_build_setup)

find_package(UsePsapp REQUIRED)
exclude_from_coverage(psapp)

find_package(Boost REQUIRED COMPONENTS python system)
include_directories(SYSTEM ${Boost_INCLUDE_DIRS})

find_package(python REQUIRED)
include_directories(SYSTEM ${Python_INCLUDE_DIR})

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

file(COPY pydbc DESTINATION .)