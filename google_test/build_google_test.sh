echo "Cloning latest google test repository"
git clone https://github.com/google/googletest.git

mkdir build
cd build
cmake ../googletest -DCMAKE_INSTALL_PREFIX=../dist
make install
