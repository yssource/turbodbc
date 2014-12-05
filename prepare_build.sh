mkdir -p ../pydbc_build
cd ../pydbc_build 
cmake -DCMAKE_MODULE_PATH="/localdata/mkoenig/workspace/cmake_scripts;/localdata/mkoenig/workspace/pydbc/" ../pydbc
ln -s ../pydbc/PyTest PyTest
