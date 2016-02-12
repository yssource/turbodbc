include_directories(SYSTEM ${Boost_INCLUDE_DIRS})

find_package(odbc REQUIRED)
include_directories(SYSTEM ${Odbc_INCLUDE_DIR})

include_directories(Library)
add_subdirectory(Library)

if(BUILD_TESTING)
    find_package(cppunit_toolbox)
    add_subdirectory(Test)
endif()
