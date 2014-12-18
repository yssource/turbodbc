mkdir -p ../pydbc_build
cd ../pydbc_build 
cmake -DCMAKE_MODULE_PATH="${BY_HOME_BASE}/workspace/cmake_scripts" -G"Eclipse CDT4 - Unix Makefiles" -DCMAKE_ECLIPSE_VERSION=4.2 -DCMAKE_ECLIPSE_GENERATE_SOURCE_PROJECT=TRUE ../pydbc
ln -s ../pydbc/PyTest PyTest
