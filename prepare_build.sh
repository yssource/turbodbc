BY_HOME_BASE=/home/localdata/${USER}
mkdir -p ../cpp_odbc_build
cd ../cpp_odbc_build 
cmake -DCMAKE_MODULE_PATH=${BY_HOME_BASE}/workspace/cmake_scripts -G"Eclipse CDT4 - Unix Makefiles" -DCMAKE_ECLIPSE_VERSION=4.2 -DCMAKE_ECLIPSE_GENERATE_SOURCE_PROJECT=TRUE ../cpp_odbc
