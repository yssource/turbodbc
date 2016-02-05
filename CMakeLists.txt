cmake_minimum_required(VERSION 2.8.7)
project(turbodbc_intern CXX)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR})

include(default_build_setup)

add_subdirectory(google_test)
add_subdirectory(cpp)
add_subdirectory(python)

if(BUILD_TESTING)
    enable_testing()
endif()

add_subdirectory(Wheel)