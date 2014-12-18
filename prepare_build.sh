mkdir -p ../pydbc_build
cd ../pydbc_build 
cmake -DCMAKE_MODULE_PATH="/localdata/klaemke/workspace/build_environment/cmake_scripts;/localdata/klaemke/workspace/git/pydbc/" ../pydbc
ln -s ../pydbc/PyTest PyTest
