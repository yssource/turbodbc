Turbodbc
========

[![Build Status](https://travis-ci.org/blue-yonder/turbodbc.svg?branch=master)](https://travis-ci.org/blue-yonder/turbodbc)

Turbodbc is a Python module to access relational databases via the Open Database
Connectivity (ODBC) interface. The module complies with the Python Database API
Specification 2.0.

Turbodbc implements both sending queries and retrieving result sets with
support for bulk operations. This allows fast inserts of large batches of
records without relying on vendor-specific mechanism such as uploads of CSV
files.

Under the Python hood, turbodbc uses several layers of C++11 code to abstract
from the low-level C API provided by the unixODBC package. This allows for
comparatively easy implementation of high-level features. 


Features
--------

*   Bulk retrieval of select statements
*   Bulk transfer of query parameters
*   Automatic conversion of decimal type to integer, float, and string as
    appropriate
*   Supported data types for both result sets and parameters:
    `int`, `float`, `str`, `bool`, `datetime.date`, `datetime.datetime`


Usage
-----

To use the latest version of turbodbc, you need to follow these steps:

*   Get the source code from github
*   Check the source build requirements (see below) are installed on your computer
*   Create a build directory. Make this your working directory.
*   Execute the following command:

        cmake -DCMAKE_INSTALL_PREFIX=./dist /path/to/source/directory

    This will prepare the build directory for the actual build step.

*   Execute the `make` command to build the code.
*   You can execute the tests with `ctest`.
*   To create a Python source distribution for simple installation, use
    the following commands:
    
        make install
        cd dist
        python setup.py sdist
    
    This will create a `.tar.gz` file in the folder `dist/dist` in your
    build directory. This file is self-contained and can be installed by
    other users using `pip install`.


Source build requirements
-------------------------

*   g++ 4.7.2+ compiler
*   boost development headers (Contained in: libboost-all-devel)
*   CMake


Supported environments
----------------------

*   Linux (successfully built on Ubuntu 12, Ubuntu 14, Debian 7)
*   Python 2.7 only (yet) 