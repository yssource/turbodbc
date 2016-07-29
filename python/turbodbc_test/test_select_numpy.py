import datetime
from collections import OrderedDict

from numpy.ma import MaskedArray
from numpy.testing import assert_equal
import numpy
import pytest

import turbodbc

from query_fixture import query_fixture
from helpers import open_cursor, for_each_database, for_one_database


def _fix_case(configuration, string):
    """
    some databases return column names in upper case
    """
    capabilities = configuration['capabilities']
    if capabilities['reports_column_names_as_upper_case']:
        return string.upper()
    else:
        return string


@for_one_database
def test_numpy_without_result_set_raises(dsn, configuration):
    with open_cursor(configuration) as cursor:
        with pytest.raises(turbodbc.InterfaceError):
            cursor.fetchallnumpy()


@for_each_database
def test_numpy_empty_column(dsn, configuration):
    with open_cursor(configuration) as cursor:
        with query_fixture(cursor, configuration, 'INSERT INTEGER') as table_name:
            cursor.execute("SELECT a FROM {}".format(table_name))
            results = cursor.fetchallnumpy()
            assert isinstance(results, OrderedDict)
            assert len(results) == 1
            assert isinstance(results[_fix_case(configuration, 'a')], MaskedArray)


@for_each_database
def test_numpy_int_column(dsn, configuration):
    with open_cursor(configuration) as cursor:
        cursor.execute("SELECT 42 AS a")
        results = cursor.fetchallnumpy()
        expected = MaskedArray([42], mask=[0])
        assert results[_fix_case(configuration, 'a')].dtype == numpy.int64
        assert_equal(results[_fix_case(configuration, 'a')], expected)


@for_each_database
def test_numpy_double_column(dsn, configuration):
    with open_cursor(configuration) as cursor:
        with query_fixture(cursor, configuration, 'SELECT DOUBLE') as query:
            cursor.execute(query)
            results = cursor.fetchallnumpy()
            expected = MaskedArray([3.14], mask=[0])
            assert results[_fix_case(configuration, 'a')].dtype == numpy.float64
            assert_equal(results[_fix_case(configuration, 'a')], expected)


@for_each_database
def test_numpy_boolean_column(dsn, configuration):
    with open_cursor(configuration) as cursor:
        with query_fixture(cursor, configuration, 'INSERT INDEXED BOOL') as table_name:
            cursor.executemany('INSERT INTO {} VALUES (?, ?)'.format(table_name),
                                [[True, 1], [False, 2], [True, 3]])
            cursor.execute('SELECT A FROM {} ORDER BY B'.format(table_name))
            results = cursor.fetchallnumpy()
            expected = MaskedArray([True, False, True], mask=[0])
            assert results[_fix_case(configuration, 'a')].dtype == numpy.bool_
            assert_equal(results[_fix_case(configuration, 'a')], expected)


@for_each_database
def test_numpy_binary_column_with_null(dsn, configuration):
    with open_cursor(configuration) as cursor:
        with query_fixture(cursor, configuration, 'INSERT TWO INTEGER COLUMNS') as table_name:
            cursor.executemany("INSERT INTO {} VALUES (?, ?)".format(table_name),
                               [[42, 1], [None, 2]]) # second column to enforce ordering
            cursor.execute("SELECT a FROM {} ORDER BY b".format(table_name))
            results = cursor.fetchallnumpy()
            expected = MaskedArray([42, 0], mask=[0, 1])
            assert_equal(results[_fix_case(configuration, 'a')], expected)


@for_each_database
def test_numpy_binary_column_larger_than_batch_size(dsn, configuration):
    with open_cursor(configuration, rows_to_buffer=2) as cursor:
        with query_fixture(cursor, configuration, 'INSERT INTEGER') as table_name:
            cursor.executemany("INSERT INTO {} VALUES (?)".format(table_name),
                               [[1], [2], [3], [4], [5]])
            cursor.execute("SELECT a FROM {} ORDER BY a".format(table_name))
            results = cursor.fetchallnumpy()
            expected = MaskedArray([1, 2, 3, 4, 5], mask=False)
            assert_equal(results[_fix_case(configuration, 'a')], expected)


@for_each_database
def test_numpy_timestamp_column(dsn, configuration):
    timestamp = datetime.datetime(2015, 12, 31, 1, 2, 3)

    with open_cursor(configuration) as cursor:
        with query_fixture(cursor, configuration, 'INSERT TIMESTAMP') as table_name:
            cursor.execute('INSERT INTO {} VALUES (?)'.format(table_name), [timestamp])
            cursor.execute('SELECT A FROM {}'.format(table_name))
            results = cursor.fetchallnumpy()
            expected = MaskedArray([timestamp], mask=[0], dtype='datetime64[us]')
            assert results[_fix_case(configuration, 'a')].dtype == numpy.dtype('datetime64[us]')
            assert_equal(results[_fix_case(configuration, 'a')], expected)

@for_each_database
def test_numpy_timestamp_column_larger_than_batch_size(dsn, configuration):
    timestamps = [datetime.datetime(2015, 12, 31, 1, 2, 3),
                  datetime.datetime(2016, 1, 5, 4, 5, 6),
                  datetime.datetime(2017, 2, 6, 7, 8, 9),
                  datetime.datetime(2018, 3, 7, 10, 11, 12),
                  datetime.datetime(3999, 4, 8, 13, 14, 15)]

    with open_cursor(configuration, rows_to_buffer=2) as cursor:
        with query_fixture(cursor, configuration, 'INSERT TIMESTAMP') as table_name:
            cursor.executemany('INSERT INTO {} VALUES (?)'.format(table_name),
                               [[timestamp] for timestamp in timestamps])
            cursor.execute('SELECT A FROM {}'.format(table_name))
            results = cursor.fetchallnumpy()
            expected = MaskedArray(timestamps, mask=[0], dtype='datetime64[us]')
            assert_equal(results[_fix_case(configuration, 'a')], expected)



@for_each_database
def test_numpy_two_columns(dsn, configuration):
    with open_cursor(configuration) as cursor:
        with query_fixture(cursor, configuration, 'INSERT TWO INTEGER COLUMNS') as table_name:
            cursor.executemany("INSERT INTO {} VALUES (?, ?)".format(table_name),
                               [[1, 42], [2, 41]])
            cursor.execute("SELECT a, b FROM {} ORDER BY a".format(table_name))
            results = cursor.fetchallnumpy()
            assert_equal(results[_fix_case(configuration, 'a')], MaskedArray([1, 2], mask=False))
            assert_equal(results[_fix_case(configuration, 'b')], MaskedArray([42, 41], mask=False))
