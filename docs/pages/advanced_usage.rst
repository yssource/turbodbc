.. _advanced_usage:

Advanced usage
==============

.. _advanced_usage_options:

Performance, compatibility, and behavior options
------------------------------------------------

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
    ...                        autocommit=True)


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

.. _advanced_usage_options_write_buffer:

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

Set ``prefer_unicode`` to ``True`` if your database does not fully support
the UTF-8 encoding turbodbc prefers. With this option you can tell turbodbc
to use two-byte character strings with UCS-2/UTF-16 encoding. Use this option
if you try to connection to Microsoft SQL server (MSSQL).


.. _advanced_usage_options_autocommit:

Autocommit
~~~~~~~~~~

Set ``autocommit`` to ``True`` if you want the database to ``COMMIT`` your
changes automatically after each query or command. By default, ``autocommit``
is disabled and users are required to call ``cursor.commit()`` to persist
their changes.

.. note::
    Some databases that do not support transactions may even require this
    option to be set to ``True`` in order to establish a connection at all.


Controlling autocommit behavior at runtime
------------------------------------------

You can enable and disable autocommit mode after you have established a connection,
and you can also check whether autocommit is currently enabled:

::

    >>> from turbodbc import connect
    >>> connection = connect(dsn="my DSN")
    >>> connection.autocommit = True

    [... more things happening ...]

    >>> if not connection.autocommit:
    ...     connection.commit()


.. _advanced_usage_numpy:

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


Notes regarding NumPy result sets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


*   NumPy results are returned as an ``OrderedDict`` of column name/value pairs. The column
    order is the same as in your query.
*   The column values are of type ``MaskedArray``. Any ``NULL`` values you have in your
    database will show up as masked entries (``NULL`` values in string-like columns
    will shop up as ``None`` objects).

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


.. _advanced_usage_numpy_parameters:

Using NumPy arrays as query parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Here is how to use turbodbc to use values stored in NumPy arrays
as query parameters with ``executemanycolumns()``:

::

    >>> from numpy import array
    >>> from numpy.ma import MaskedArray
    >>> normal_param = array([1, 2, 3], dtype='int64')
    >>> masked_param = MaskedArray([3.14, 1.23, 4.56], mask=[False, True, False], dtype='float64')

    >>> cursor.executemanycolumns("INSERT INTO my_table VALUES (?, ?)", [normal_param, masked_param])
    # functionally equivalent, but much faster than:
    # cursor.execute("INSERT INTO my_table VALUES (1, 3.14)")
    # cursor.execute("INSERT INTO my_table VALUES (2, NULL)")
    # cursor.execute("INSERT INTO my_table VALUES (3, 4.56)")

    >>> cursor.execute("SELECT * FROM my_table").fetchall()
    [[1L, 3.14], [2L, None], [3L, 4.56]]

*   Columns must either be of type ``MaskedArray`` or ``ndarray``.
*   Each column must contain one-dimensional, contiguous data.
*   All columns must have equal size.
*   The ``dtype`` of each column must be supported, see the table below.
*   Use ``MaskedArray``s with and set the ``mask`` to ``True`` for individual
    elements to use ``None`` values.
*   Data is transfered in batches (see :ref:`advanced_usage_options_write_buffer`)


+-------------------------------------------------------------------------+--------------------------------+
| Supported NumPy type                                                    | Transfered as                  |
+=========================================================================+================================+
| ``int64``                                                               | ``BIGINT`` (64 bits)           |
+-------------------------------------------------------------------------+--------------------------------+
| ``float64``                                                             | ``DOUBLE PRECISION`` (64 bits) |
+-------------------------------------------------------------------------+--------------------------------+
| ``bool_``                                                               | ``BIT``                        |
+-------------------------------------------------------------------------+--------------------------------+
| ``datetime64[us]``                                                      | ``TIMESTAMP``                  |
+-------------------------------------------------------------------------+--------------------------------+
| ``datetime64[D]``                                                       | ``DATE``                       |
+-------------------------------------------------------------------------+--------------------------------+
| ``object_`` (only ``str``, ``unicode``, and ``None`` objects supported) | ``VARCHAR`` (automatic sizing) |
+-------------------------------------------------------------------------+--------------------------------+

.. _advanced_usage_arrow:

Apache Arrow support
--------------------

.. note::
    Turbodbc's Apache Arrow support requires the ``pyarrow`` package to be installed.
    For all source builds, Apache Arrow needs to be installed before installing turbodbc.
    Please check the :ref:`installation instructions <getting_started_installation>`
    for more details.

`Apache Arrow <https://arrow.apache.org>`_ is a high-performance data layer that
is built for cross-system columnar in-memory analytics using a
`data model <https://arrow.apache.org/docs/python/data.html>`_ designed to make the
most of the CPU cache and vector operations.

.. note::
    Apache Arrow support in turbodbc is still experimental and may not be as efficient
    as possible yet. Also, Apache Arrow support is not yet available for Windows and
    has some issues with Unicode fields. Stay tuned for upcoming improvements.

Obtaining Apache Arrow result sets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Here is how to use turbodbc to retrieve the full result set in the form of an
Apache Arrow table:

::

    >>> cursor.execute("SELECT A, B FROM my_table")
    >>> table = cursor.fetchallarrow()
    >>> table
    pyarrow.Table
    A: int64
    B: string
    >>> table[0].to_pylist()
    [42]
    >>> table[1].to_pylist()
    [u'hello']

Looking at the data like this is not particularly useful. However, there is some
really useful stuff you can do with an Apache Arrow table, for example,
`convert it to a Pandas dataframe <https://arrow.apache.org/docs/python/pandas.html>`_
like this:

::

    >>> table.to_pandas()
        A      B
    0  42  hello
