from numpy.ma import MaskedArray
from numpy import array

import pytest

import turbodbc

from query_fixture import query_fixture
from helpers import open_cursor, for_one_database


@for_one_database
def test_column_of_unsupported_type_raises(dsn, configuration):
    with open_cursor(configuration) as cursor:
        with query_fixture(cursor, configuration, 'INSERT INTEGER') as table_name:
            columns = ["this is not a NumPy MaskedArray"]
            with pytest.raises(turbodbc.InterfaceError):
                cursor.executemanycolumns("INSERT INTO {} VALUES (?)".format(table_name), columns)


@for_one_database
def test_columns_of_unequal_sizes_raise(dsn, configuration):
    with open_cursor(configuration) as cursor:
        with query_fixture(cursor, configuration, 'INSERT INTEGER') as table_name:
            columns = [MaskedArray([1, 2, 3], mask=False, dtype='int64'),
                       MaskedArray([1, 2], mask=False, dtype='int64')]
            with pytest.raises(turbodbc.InterfaceError):
                cursor.executemanycolumns("INSERT INTO {} VALUES (?)".format(table_name), columns)


@for_one_database
def test_column_with_incompatible_dtype_raises(dsn, configuration):
    with open_cursor(configuration) as cursor:
        with query_fixture(cursor, configuration, 'INSERT INTEGER') as table_name:
            columns = [MaskedArray([1, 2, 3], mask=False, dtype='int16')]
            with pytest.raises(turbodbc.InterfaceError):
                cursor.executemanycolumns("INSERT INTO {} VALUES (?)".format(table_name), columns)


@for_one_database
def test_column_with_multiple_dimensions_raises(dsn, configuration):
    with open_cursor(configuration) as cursor:
        with query_fixture(cursor, configuration, 'INSERT INTEGER') as table_name:
            two_dimensional = array([[1, 2, 3], [4, 5, 6]], dtype='int64')
            columns = [two_dimensional]
            with pytest.raises(turbodbc.InterfaceError):
                cursor.executemanycolumns("INSERT INTO {} VALUES (?)".format(table_name), columns)


@for_one_database
def test_column_with_non_contiguous_data_raises(dsn, configuration):
    with open_cursor(configuration) as cursor:
        with query_fixture(cursor, configuration, 'INSERT INTEGER') as table_name:
            two_dimensional = array([[1, 2, 3], [4, 5, 6]], dtype='int64')
            one_dimensional = two_dimensional[:, 1]
            assert one_dimensional.flags.c_contiguous == False
            columns = [one_dimensional]
            with pytest.raises(turbodbc.InterfaceError):
                cursor.executemanycolumns("INSERT INTO {} VALUES (?)".format(table_name), columns)


@for_one_database
def test_number_of_columns_does_not_match_parameter_count(dsn, configuration):
    with open_cursor(configuration) as cursor:
        with query_fixture(cursor, configuration, 'INSERT INTEGER') as table_name:
            columns = [array([42], dtype='int64'), array([17], dtype='int64')]
            with pytest.raises(turbodbc.InterfaceError):
                cursor.executemanycolumns("INSERT INTO {} VALUES (?)".format(table_name), columns)


@for_one_database
def test_passing_empty_list_of_columns_is_ok(dsn, configuration):
    with open_cursor(configuration) as cursor:
        with query_fixture(cursor, configuration, 'INSERT INTEGER') as table_name:
            cursor.executemanycolumns("INSERT INTO {} VALUES (42)".format(table_name), [])

            results = cursor.execute("SELECT A FROM {} ORDER BY A".format(table_name)).fetchall()
            assert results == [[42]]


@for_one_database
def test_passing_empty_column_is_ok(dsn, configuration):
    with open_cursor(configuration) as cursor:
        with query_fixture(cursor, configuration, 'INSERT INTEGER') as table_name:
            columns = [array([], dtype='int64')]
            cursor.executemanycolumns("INSERT INTO {} VALUES (?)".format(table_name), columns)

            results = cursor.execute("SELECT A FROM {} ORDER BY A".format(table_name)).fetchall()
            assert results == []
