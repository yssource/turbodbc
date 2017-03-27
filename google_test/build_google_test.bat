echo "Cloning latest google test repository"
git clone https://github.com/google/googletest.git

md build
cd build
cmake ../googletest -DCMAKE_INSTALL_PREFIX=../dist -DCMAKE_VERBOSE_MAKEFILE=ON -A x64
cmake --build . --target install
