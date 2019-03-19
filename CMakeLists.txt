cmake_minimum_required(VERSION 2.8.12)
project(turbodbc_intern CXX)

# Take precedence over system modules
list(INSERT CMAKE_MODULE_PATH 0 ${CMAKE_SOURCE_DIR}/cmake_scripts)

include(default_build_setup)

add_subdirectory(pybind11)
find_package(GTest)
if (GTest_FOUND OR GTEST_FOUND)
  include_directories(${GTEST_INCLUDE_DIR})
else()
  add_subdirectory(google_test)
  set(GTEST_LIBRARY gtest)
  set(GMOCK_LIBRARY gmock)
  set(GMOCK_MAIN_LIBRARY gmock_main)
endif()
add_subdirectory(cpp)
add_subdirectory(python)
add_subdirectory(docs)

if(BUILD_TESTING)
    enable_testing()
endif()

install(
	FILES setup.py setup.cfg MANIFEST.in Makefile LICENSE README.md CHANGELOG.rst
	DESTINATION .
)
