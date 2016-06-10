Turbodbc
========

[![Build Status](https://travis-ci.org/blue-yonder/turbodbc.svg?branch=master)](https://travis-ci.org/blue-yonder/turbodbc)

Turbodbc is a Python module to access relational databases via the Open Database
Connectivity (ODBC) interface. The module complies with the Python Database API
Specification 2.0.


Why should I use turbodbc instead of other ODBC libraries?
----------------------------------------------------------

Short answer: turbodbc is faster.

Slightly longer answer: I have tested turbodbc and pyodbc (probably the most
popular Python ODBC module) with various databases (Exasol, PostgreSQL, MySQL)
and corresponding ODBC drivers. I found turbodbc to be consistently faster.

For retrieving result sets, I found speedups between 1.5 and 7. For inserting
data, I found speedups of up to 100.


Smooth. What is the trick?
--------------------------

There is not really a trick. Turbodbc implements both sending parameters and
retrieving result sets using buffers of multiple rows/parameter sets. This
avoids round trips to the ODBC driver and (depending how well the ODBC driver
is written) to the database. 


Features
--------

*   Bulk retrieval of select statements
*   Bulk transfer of query parameters
*   Automatic conversion of decimal type to integer, float, and string as
    appropriate
*   Supported data types for both result sets and parameters:
    `int`, `float`, `str`, `bool`, `datetime.date`, `datetime.datetime`
*   Also provides a high-level C++11 database driver under the hood


Installation
------------

To install turbodbc, please make sure you have the following things installed:

*   A modern g++ compiler (works with 4.7.2+)
*   Boost development headers (typical package name: libboost-all-devel)
*   Unixodbc development headers
*   Python 2.7 development headers

To install turbodbc, please use the following command:

    pip install turbodbc


Basic usage
-----------

Turbodbc follows the specification of the Python database API v2, which you can
find at https://www.python.org/dev/peps/pep-0249/. Here is a short summary,
including the parts not specified.

To establish a connection, use any of the following commands:

    >>> from turbodbc import connect
    >>> connection = connect(dsn='My data source name as given by odbc.ini')
    >>> connection = connect(dsn='my dsn', user='my user has precedence')
    >>> connection = connect(dsn='my dsn', username='field names depend on the driver')

To execute a query, you need a `cursor` object:

    >>> cursor = connection.cursor()

Here is how to execute a `SELECT` query:

    >>> cursor.execute('SELECT 42')
    >>> for row in cursor:
    >>>     print list(row)

Here is how to execute an `INSERT` query with many parameters:

    >>> parameter_sets = [['hi', 42],
                          ['there', 23]]
    >>> cursor.executemany('INSERT INTO my_table VALUES (?, ?)',
                           parameter_sets)


Development version
-------------------

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

For the development build, you also require the following additional 
dependencies:

*   CMake


Supported environments
----------------------

*   Linux (successfully built on Ubuntu 12, Ubuntu 14, Debian 7)
*   Python 2.7 only (yet) 