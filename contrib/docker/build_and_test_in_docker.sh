#!/bin/bash

set -ex

pushd /io

export ODBCSYSINI=`pwd`/travis/odbc
export TURBODBC_TEST_CONFIGURATION_FILES="query_fixtures_postgresql.json,query_fixtures_mysql.json"

/etc/init.d/mysql start
mysql -u root -e 'CREATE DATABASE test_db;'

/etc/init.d/postgresql start
sudo -u postgres psql -U postgres -c 'CREATE DATABASE test_db;'
sudo -u postgres psql -U postgres -c "ALTER USER postgres WITH PASSWORD 'password';"

mkdir -p build_docker && cd build_docker
cmake -DCMAKE_INSTALL_PREFIX=./dist -DPYBIND11_PYTHON_VERSION=2.7 ..
make -j5
ctest --verbose
