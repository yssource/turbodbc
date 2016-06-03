Version 0.2.3
=============

*   Fix issue that only lists were allowed for specifying parameters for queries
*   Improve parameter memory consumption when the database reports very large
    string parameter sizes 
*   Internal: Provides more low-level ways to access the result set

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