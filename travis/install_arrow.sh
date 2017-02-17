#!/bin/bash -xe

wget https://dist.apache.org/repos/dist/dev/arrow/apache-arrow-0.2.0-rc2/apache-arrow-0.2.0.tar.gz
tar xf apache-arrow-0.2.0.tar.gz
pushd apache-arrow-0.2.0/

export ARROW_HOME=$TRAVIS_BUILD_DIR/dist

mkdir cpp/build
pushd cpp/build
cmake -DCMAKE_INSTALL_PREFIX=$ARROW_HOME ..
make -j3
make install
popd

pushd python
python setup.py build_ext --inplace
python setup.py install --single-version-externally-managed --record record.txt
popd
