from collections import OrderedDict

from numpy.ma import MaskedArray
import pytest

import turbodbc

from query_fixture import query_fixture
from helpers import open_cursor, for_each_database, for_one_database


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
            assert isinstance(results['A'], MaskedArray)

# @for_each_database
# def test_numpy_int_column(dsn, configuration):
#     with open_cursor(configuration) as cursor:
#         cursor.execute("SELECT 42 AS a")
#         results = cursor.fetchallnumpy()
#         expected = MaskedArray([42], mask=[0])
#         assert results['A'] == expected