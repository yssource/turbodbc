find_package(Boost REQUIRED COMPONENTS locale)

find_package(odbc REQUIRED)
include_directories(SYSTEM ${Odbc_INCLUDE_DIR})

include_directories(Library)
add_subdirectory(Library)

if(BUILD_TESTING)
    add_subdirectory(Test)
endif()
