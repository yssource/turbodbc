import datetime

from numpy.ma import MaskedArray
from numpy import array

from query_fixture import query_fixture
from helpers import open_cursor, for_each_database, generate_microseconds_with_precision


def _test_basic_column(configuration, fixture, values, dtype):
    with open_cursor(configuration) as cursor:
        with query_fixture(cursor, configuration, fixture) as table_name:
            columns = [array(values, dtype=dtype)]
            cursor.executemanycolumns("INSERT INTO {} VALUES (?)".format(table_name), columns)

            results = cursor.execute("SELECT A FROM {} ORDER BY A".format(table_name)).fetchall()
            assert results == [[value] for value in sorted(values)]


def _test_column_exceeds_buffer_size(configuration, fixture, values, dtype):
    with open_cursor(configuration, parameter_sets_to_buffer=2) as cursor:
        with query_fixture(cursor, configuration, fixture) as table_name:
            columns = [array(values, dtype=dtype)]
            cursor.executemanycolumns("INSERT INTO {} VALUES (?)".format(table_name), columns)

            results = cursor.execute("SELECT A FROM {} ORDER BY A".format(table_name)).fetchall()
            assert results == [[value] for value in sorted(values)]


def _test_masked_column(configuration, fixture, values, dtype):
    with open_cursor(configuration) as cursor:
        with query_fixture(cursor, configuration, fixture) as table_name:
            columns = [MaskedArray(values, mask=[False, True, False], dtype=dtype)]
            cursor.executemanycolumns("INSERT INTO {} VALUES (?)".format(table_name), columns)

            results = cursor.execute("SELECT A FROM {} ORDER BY A".format(table_name)).fetchall()
            assert results == [[value] for value in [values[0], values[2]]] + [[None]] or \
                results == [[None]] + [[value] for value in [values[0], values[2]]]


def _test_masked_column_with_shrunk_mask(configuration, fixture, values, dtype):
    with open_cursor(configuration) as cursor:
        with query_fixture(cursor, configuration, fixture) as table_name:
            columns = [MaskedArray(values, mask=False, dtype=dtype)]
            columns[0].shrink_mask()
            cursor.executemanycolumns("INSERT INTO {} VALUES (?)".format(table_name), columns)

            results = cursor.execute("SELECT A FROM {} ORDER BY A".format(table_name)).fetchall()
            assert results == [[value] for value in sorted(values)]


def _test_masked_column_exceeds_buffer_size(configuration, fixture, values, dtype):
    with open_cursor(configuration, parameter_sets_to_buffer=2) as cursor:
        with query_fixture(cursor, configuration, fixture) as table_name:
            columns = [MaskedArray(values, mask=[True, False, True], dtype=dtype)]
            cursor.executemanycolumns("INSERT INTO {} VALUES (?)".format(table_name), columns)

            results = cursor.execute("SELECT A FROM {} ORDER BY A".format(table_name)).fetchall()
            assert results == [[values[1]], [None], [None]] or \
                results == [[None], [None], [values[1]]]


def _test_single_masked_value(configuration, fixture, values, dtype):
    with open_cursor(configuration) as cursor:
        with query_fixture(cursor, configuration, fixture) as table_name:
            columns = [MaskedArray([values[0]], mask=[True], dtype=dtype)]
            cursor.executemanycolumns("INSERT INTO {} VALUES (?)".format(table_name), columns)

            results = cursor.execute("SELECT A FROM {} ORDER BY A".format(table_name)).fetchall()
            assert results == [[None]]


def _full_column_tests(configuration, fixture, values, dtype):
    _test_basic_column(configuration, fixture, values, dtype)
    _test_column_exceeds_buffer_size(configuration, fixture, values, dtype)
    _test_masked_column(configuration, fixture, values, dtype)
    _test_masked_column_with_shrunk_mask(configuration, fixture, values, dtype)
    _test_masked_column_exceeds_buffer_size(configuration, fixture, values, dtype)
    _test_single_masked_value(configuration, fixture, values, dtype)


@for_each_database
def test_integer_column(dsn, configuration):
    _full_column_tests(configuration, "INSERT INTEGER", [17, 23, 42], 'int64')


@for_each_database
def test_float64_column(dsn, configuration):
    _full_column_tests(configuration, "INSERT DOUBLE", [2.71, 3.14, 6.25], 'float64')


@for_each_database
def test_datetime64_us_column(dsn, configuration):
    supported_digits = configuration['capabilities']['fractional_second_digits']
    fractional = generate_microseconds_with_precision(supported_digits)

    _full_column_tests(configuration,
                       "INSERT TIMESTAMP",
                       [datetime.datetime(2015, 12, 31, 1, 2, 3, fractional),
                        datetime.datetime(2016, 1, 1, 4, 5, 6, fractional),
                        datetime.datetime(2017, 5, 6, 7, 8, 9, fractional)],
                       'datetime64[us]')



@for_each_database
def test_multiple_columns(dsn, configuration):
    with open_cursor(configuration) as cursor:
        with query_fixture(cursor, configuration, 'INSERT TWO INTEGER COLUMNS') as table_name:
            columns = [array([17, 23, 42], dtype='int64'), array([3, 2, 1], dtype='int64')]
            cursor.executemanycolumns("INSERT INTO {} VALUES (?, ?)".format(table_name), columns)

            results = cursor.execute("SELECT A, B FROM {} ORDER BY A".format(table_name)).fetchall()
            assert results == [[17, 3], [23, 2], [42, 1]]
