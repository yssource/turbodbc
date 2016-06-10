Version 0.2.4
=============

*   Improve performance of Python object conversion while reading result sets.
    In tests with an Exasol database, performance got about 15% better.
*   C++ backend: `turbodbc::cursor` no longer allows direct access to the C++
    `field` type. Instead, please use the `cursor`'s `get_query()` method,
    and construct a `turbodbc::result_sets::field_result_set` using the
    `get_results()` method.

Version 0.2.3
=============

*   Fix issue that only lists were allowed for specifying parameters for queries
*   Improve parameter memory consumption when the database reports very large
    string parameter sizes 
*   C++ backend: Provides more low-level ways to access the result set

Version 0.2.2
=============

*   Fix issue that `dsn` parameter was always present in the connection string
    even if it was not set by the user's call to `connect()`
*   Internal: First version to run on Travis.
*   Internal: Use pytest instead of unittest for testing
*   Internal: Allow for integration tests to run in custom environment
*   Internal: Simplify integration test configuration


Version 0.2.1
=============

*   Internal: Change C++ test framework to Google Test


Version 0.2.0
=============

*   New parameter types supported: `bool`, `datetime.date`, `datetime.datetime`
*   `cursor.rowcount` returns number of affected rows for manipulating queries
*   `Connection` supports `rollback()`
*   Improved handling of string parameters


Version 0.1.0
=============

Initial release