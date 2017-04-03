echo "Cloning latest google test repository"
git clone https://github.com/google/googletest.git

md build
cd build
cmake ../googletest -DCMAKE_INSTALL_PREFIX=../dist -DCMAKE_VERBOSE_MAKEFILE=ON -DCMAKE_BUILD_TYPE=Release -A x64
cmake --build . --config Release --target install
