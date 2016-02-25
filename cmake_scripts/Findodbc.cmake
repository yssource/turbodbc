# This script provides include and link directories for unixODBC

message(STATUS "Detecting unixODBC library")

find_path(
    Odbc_INCLUDE_DIR
    sql.h
    HINTS
        ENV UNIXODBC_INCLUDE_DIR
    
    DOC "Path to the unixODBC headers"
)

if("${Odbc_INCLUDE_DIR}" STREQUAL "Odbc_INCLUDE_DIR-NOTFOUND")
    message(SEND_ERROR " Could not find unixODBC header files")
else()
    message(STATUS "  Found header files at: ${Odbc_INCLUDE_DIR}")
endif()


find_library(
    Odbc_LIBRARIES
    odbc
    HINTS        
        ENV UNIXODBC_LIB_DIR
           
    DOC "The unixODBC library"
)

if("${Odbc_LIBRARIES}" STREQUAL "Odbc_LIBRARIES-NOTFOUND")
    message(FATAL_ERROR " Could not find unixODBC library")
else()
    message(STATUS "  Found library at: ${Odbc_LIBRARIES}")
endif()

get_filename_component(Odbc_LIBRARY_DIRS "${Odbc_LIBRARIES}" PATH)
