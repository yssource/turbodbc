mkdir -p ../pydbc_build
cd ../pydbc_build 
cmake -DCMAKE_MODULE_PATH="/localdata/mkoenig/workspace/cmake_scripts;/localdata/mkoenig/workspace/pydbc/" -G"Eclipse CDT4 - Unix Makefiles" -DCMAKE_ECLIPSE_VERSION=4.2 -DCMAKE_ECLIPSE_GENERATE_SOURCE_PROJECT=TRUE ../pydbc
ln -s ../pydbc/PyTest PyTest
