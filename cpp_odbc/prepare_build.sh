BY_HOME_BASE=/localdata/${USER}
mkdir -p ../cpp_odbc_build
cd ../cpp_odbc_build 
cmake -DCMAKE_MODULE_PATH=${BY_HOME_BASE}/workspace/build_tools/cmake_scripts -G"Eclipse CDT4 - Unix Makefiles" -DCMAKE_ECLIPSE_VERSION=4.2 -DCMAKE_CXX_COMPILER_ARG1=-std=c++11 -DCMAKE_ECLIPSE_GENERATE_SOURCE_PROJECT=TRUE ../cpp_odbc
