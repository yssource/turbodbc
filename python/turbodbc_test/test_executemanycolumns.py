import datetime
from collections import OrderedDict

from numpy.ma import MaskedArray
from numpy.testing import assert_equal
import numpy
import pytest

import turbodbc

from query_fixture import query_fixture
from helpers import open_cursor, for_each_database, for_one_database


@for_one_database
def test_column_must_be_masked_array(dsn, configuration):
    with open_cursor(configuration) as cursor:
        with query_fixture(cursor, configuration, 'INSERT INTEGER') as table_name:
            columns = ["this is not a NumPy MaskedArray"]
            with pytest.raises(turbodbc.InterfaceError):
                cursor.executemanycolumns("INSERT INTO {} VALUES (?)".format(table_name), columns)


@for_one_database
def test_columns_must_have_equal_size(dsn, configuration):
    with open_cursor(configuration) as cursor:
        with query_fixture(cursor, configuration, 'INSERT INTEGER') as table_name:
            columns = [MaskedArray([1, 2, 3], mask=False),
                       MaskedArray([1, 2], mask=False)]
            with pytest.raises(turbodbc.InterfaceError):
                cursor.executemanycolumns("INSERT INTO {} VALUES (?)".format(table_name), columns)


@for_one_database
def test_column_with_incompatible_dtype(dsn, configuration):
    with open_cursor(configuration) as cursor:
        with query_fixture(cursor, configuration, 'INSERT INTEGER') as table_name:
            columns = [MaskedArray([1, 2, 3], mask=False, dtype='int16')]
            with pytest.raises(turbodbc.InterfaceError):
                cursor.executemanycolumns("INSERT INTO {} VALUES (?)".format(table_name), columns)


@for_each_database
def test_integer_column(dsn, configuration):
    with open_cursor(configuration) as cursor:
        with query_fixture(cursor, configuration, 'INSERT INTEGER') as table_name:
            columns = [MaskedArray([17, 23, 42], mask=False)]
            cursor.executemanycolumns("INSERT INTO {} VALUES (?)".format(table_name), columns)

            results = cursor.execute("SELECT A FROM {} ORDER BY A".format(table_name)).fetchall()
            assert results == [[17], [23], [42]]
