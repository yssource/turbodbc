.. _advanced_usage:

Advanced usage
==============

.. _advanced_usage_options:

Performance and compatibility options
-------------------------------------

Turbodbc offers a way to adjust its behavior to tune performance and to
achieve compatibility with your database. The basic usage is this:

::

    >>> from turbodbc import connect, make_options
    >>> options = make_options()
    >>> connect(dsn="my_dsn", turbodbc_options=options)

This will connect with your database using the default options. To use non-default
options, supply keyword arguments to ``make_options()``:

::

    >>> from turbodbc import Megabytes
    >>> options = make_options(read_buffer_size=Megabytes(100),
    ...                        parameter_sets_to_buffer=1000,
    ...                        use_async_io=True,
    ...                        prefer_unicode=True)


.. _advanced_usage_options_read_buffer:

Read buffer size
~~~~~~~~~~~~~~~~

``read_buffer_size`` affects how many result set rows are retrieved per batch
of results. Set the attribute to ``turbodbc.Megabytes(42)`` to have turbodbc determine
the optimal number of rows per batch so that the total buffer amounts to
42 MB. This is recommended for most users and databases. You can also set
the attribute to ``turbodbc.Rows(13)`` if you would like to fetch results in
batches of 13 rows. By default, turbodbc fetches results in batches of 20 MB.

Please note that sometimes a single row of a result set may exceed the specified
buffer size. This can happen if large fields such as ``VARCHAR(8000000)`` or ``TEXT``
are part of the result set. In this case, results are fetched in batches of single rows
that exceed the specified size.


Buffered parameter sets
~~~~~~~~~~~~~~~~~~~~~~~

Similarly, ``parameter_sets_to_buffer`` changes the number of parameter sets
which are transferred per batch of parameters (e.g., as sent with ``executemany()``).
Please note that it is not (yet) possible to use the `Megabytes` and `Rows` classes
here.


Asynchronous input/output
~~~~~~~~~~~~~~~~~~~~~~~~~

If you set ``use_async_io`` to ``True``, turbodbc will use asynchronous I/O operations
(limited to result sets for the time being). Asynchronous I/O means that while the
main thread converts result set rows retrieved from the database to Python
objects, another thread fetches a new batch of results from the database in the background. This may yield
a speedup of ``2`` if retrieving and converting are similarly fast
operations.

.. note::
    Asynchronous I/O is experimental and has to fully prove itself yet.
    Do not be afraid to give it a try, though.


.. _advanced_usage_options_prefer_unicode:

Prefer unicode
~~~~~~~~~~~~~~

Finally, set ``prefer_unicode`` to ``True`` if your database does not fully support
the UTF-8 encoding turbodbc prefers. With this option you can tell turbodbc
to use two-byte character strings with UCS-2/UTF-16 encoding. Use this option
if you try to connection to Microsoft SQL server (MSSQL).



NumPy support
-------------

.. note::
    Turbodbc's NumPy support requires the ``numpy`` package to be installed. For all source builds,
    Numpy needs to be installed before installing turbodbc.
    Please check the :ref:`installation instructions <getting_started_installation>`
    for more details.


Obtaining NumPy result sets all at once
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Here is how to use turbodbc to retrieve the full result set in the form of NumPy
masked arrays:

::

    >>> cursor.execute("SELECT A, B FROM my_table")
    >>> cursor.fetchallnumpy()
    OrderedDict([('A', masked_array(data = [42 --],
                                    mask = [False True],
                                    fill_value = 999999)),
                 ('B', masked_array(data = [3.14 2.71],
                                    mask = [False False],
                                    fill_value = 1e+20))])


Obtaining NumPy result sets in batches
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can also fetch NumPy result sets in batches using an iterable:

::

    >>> cursor.execute("SELECT A, B FROM my_table")
    >>> batches = cursor.fetchnumpybatches()
    >>> for batch in batches:
    ...     print(batch)
    OrderedDict([('A', masked_array(data = [42 --],
                                    mask = [False True],
                                    fill_value = 999999)),
                 ('B', masked_array(data = [3.14 2.71],
                                    mask = [False False],
                                    fill_value = 1e+20))])

The size of the batches depends on the ``read_buffer_size`` attribute set in
the :ref:`performance options <advanced_usage_options_read_buffer>`.


Notes regarding NumPy support
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


*   NumPy results are returned as an ``OrderedDict`` of column name/value pairs. The column
    order is the same as in your query.
*   The column values are of type ``MaskedArray``. Any ``NULL`` values you have in your
    database will show up as masked entries (``NULL`` values in string-like columns
    will shop up as ``None`` objects).
*   NumPy support is currently limited to result sets.

The following table shows how the most common data types data scientists are interested in
are converted to NumPy columns:

+-------------------------------------------+-----------------------+
| Database type(s)                          | Python type           |
+===========================================+=======================+
| Integers, ``DECIMAL(<19,0)``              | ``int64``             |
+-------------------------------------------+-----------------------+
| ``DOUBLE``, ``DECIMAL(x, >0)``            | ``float64``           |
+-------------------------------------------+-----------------------+
| ``BIT``, boolean-like                     | ``bool_``             |
+-------------------------------------------+-----------------------+
| ``TIMESTAMP``, ``TIME``                   | ``datetime64[us]``    |
+-------------------------------------------+-----------------------+
| ``DATE``                                  | ``datetime64[D}``     |
+-------------------------------------------+-----------------------+
| ``VARCHAR``, strings, ``DECIMAL(>18, 0)`` | ``object_``           |
+-------------------------------------------+-----------------------+
