#!/bin/bash -xe

postgres psql -U postgres -c 'CREATE DATABASE test_db;'
postgres psql -U postgres -c "ALTER USER postgres WITH PASSWORD 'password';"
mysql -e 'CREATE DATABASE test_db;'
