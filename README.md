Turbodbc
========

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
*   Supported data types for result sets: `int`, `float`, `str`, `bool`,
    `datetime.date`, `datetime.datetime`
*   Supported data types for parameters: `int`, `float`, `str`