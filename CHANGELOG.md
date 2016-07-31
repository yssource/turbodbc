Version 0.4.0
=============

*   NumPy support is introduced to turbodbc for retrieving result sets.
    Use `cursor.fetchallnumpy` to retrieve a result set as an `OrderedDict`
    of `column_name: column_data` pairs, where `column_data` is a NumPy `MaskedArray`
    of appropriate type.
*   Internal: Single `turbodbc_intern` library was split up into three libraries
    to keep NumPy support optional. A few files were moved because of this.

Version 0.3.0
=============

*   turbodbc now supports asynchronous I/O operations for retrieving result sets.
    This means that while the main thread is busy converting an already retrieved
    batch of results to Python objects, another thread fetches an additional
    batch in the background. This may yield substantial performance improvements
    in the right circumstances (results are retrieved in roughly the same speed
    as they are converted to Python objects).
    
    Ansynchronous I/O support is experimental. Enable it with
    `turbodbc.connect('My data source name', use_async_io=True)`

Version 0.2.5
=============

*   C++ backend: `turbodbc::column` no longer automatically binds on
    construction. Call `bind()` instead.

Version 0.2.4
=============

*   Result set rows are returned as native Python lists instead of a not easily
    printable custom type.
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