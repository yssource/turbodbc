find_package(Boost REQUIRED COMPONENTS python system)
include_directories(SYSTEM ${Boost_INCLUDE_DIRS})

find_package(PythonLibs 2.7 REQUIRED)
include_directories(SYSTEM ${PYTHON_INCLUDE_DIRS})

find_package(odbc REQUIRED)
include_directories(SYSTEM ${Odbc_INCLUDE_DIR})

find_package(numpy REQUIRED)
include_directories(SYSTEM ${Numpy_INCLUDE_DIR})

include_directories(Library)
include_directories(../cpp_odbc/Library)
include_directories(../turbodbc/Library)
add_subdirectory(Library)
