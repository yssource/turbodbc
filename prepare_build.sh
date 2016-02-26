mkdir -p ../pydbc_build
cd ../pydbc_build 
cmake -G"Eclipse CDT4 - Unix Makefiles" -DCMAKE_ECLIPSE_VERSION=4.2 -DCMAKE_CXX_COMPILER_ARG1=-std=c++11 -DCMAKE_ECLIPSE_GENERATE_SOURCE_PROJECT=TRUE ../pydbc
