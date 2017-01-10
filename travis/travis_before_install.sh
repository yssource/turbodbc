#!/bin/bash -xe
sudo apt-get install -y python-pytest \
        unixodbc unixodbc-dev \
        libboost-date-time-dev \
        libboost-system-dev \
        mysql-server-5.6 mysql-client-core-5.6 mysql-client-5.6 libmyodbc \
        postgresql odbc-postgresql=1:09.02.0100-2ubuntu1
sudo pip install mock numpy==1.8.0 pybind11==1.8.1
