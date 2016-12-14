#!/bin/bash -xe
sudo -u postgres psql -U postgres -c 'CREATE DATABASE test_db;'
sudo -u postgres psql -U postgres -c "ALTER USER postgres WITH PASSWORD 'password';"
mysql -u root -e 'CREATE DATABASE test_db;'
