# This script provides the currently set cpp_odbc environment variables as
# include and link directories

include_directories(SYSTEM $ENV{CPP_ODBC_INCLUDE_DIR})
link_directories($ENV{CPP_ODBC_LIB_DIR})