#!/usr/bin/env bash
FLAGS=$1

echo "Cloning latest google test repository"
git clone https://github.com/google/googletest.git

mkdir build
cd build
cmake ../googletest -DCMAKE_INSTALL_PREFIX=../dist -DCMAKE_CXX_FLAGS="${FLAGS}"
cmake --build . --target install
