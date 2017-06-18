from numpy.ma import MaskedArray
from numpy import array

from query_fixture import query_fixture
from helpers import open_cursor, for_each_database


@for_each_database
def test_integer_column(dsn, configuration):
    with open_cursor(configuration) as cursor:
        with query_fixture(cursor, configuration, 'INSERT INTEGER') as table_name:
            columns = [array([17, 23, 42], dtype='int64')]
            cursor.executemanycolumns("INSERT INTO {} VALUES (?)".format(table_name), columns)

            results = cursor.execute("SELECT A FROM {} ORDER BY A".format(table_name)).fetchall()
            assert results == [[17], [23], [42]]


@for_each_database
def test_integer_column_exceeds_buffer_size(dsn, configuration):
    with open_cursor(configuration, parameter_sets_to_buffer=2) as cursor:
        with query_fixture(cursor, configuration, 'INSERT INTEGER') as table_name:
            columns = [array([17, 23, 42], dtype='int64')]
            cursor.executemanycolumns("INSERT INTO {} VALUES (?)".format(table_name), columns)

            results = cursor.execute("SELECT A FROM {} ORDER BY A".format(table_name)).fetchall()
            assert results == [[17], [23], [42]]


@for_each_database
def test_masked_integer_column(dsn, configuration):
    with open_cursor(configuration) as cursor:
        with query_fixture(cursor, configuration, 'INSERT INTEGER') as table_name:
            columns = [MaskedArray([17, 23, 42], mask=[False, True, False], dtype='int64')]
            cursor.executemanycolumns("INSERT INTO {} VALUES (?)".format(table_name), columns)

            results = cursor.execute("SELECT A FROM {} ORDER BY A".format(table_name)).fetchall()
            assert results == [[17], [42], [None]]


@for_each_database
def test_masked_integer_column_with_shrunk_mask(dsn, configuration):
    with open_cursor(configuration) as cursor:
        with query_fixture(cursor, configuration, 'INSERT INTEGER') as table_name:
            columns = [MaskedArray([17, 23, 42], mask=False, dtype='int64')]
            cursor.executemanycolumns("INSERT INTO {} VALUES (?)".format(table_name), columns)

            results = cursor.execute("SELECT A FROM {} ORDER BY A".format(table_name)).fetchall()
            assert results == [[17], [23], [42]]


@for_each_database
def test_masked_integer_column_exceeds_buffer_size(dsn, configuration):
    with open_cursor(configuration, parameter_sets_to_buffer=2) as cursor:
        with query_fixture(cursor, configuration, 'INSERT INTEGER') as table_name:
            columns = [MaskedArray([17, 23, 42], mask=[True, False, True], dtype='int64')]
            cursor.executemanycolumns("INSERT INTO {} VALUES (?)".format(table_name), columns)

            results = cursor.execute("SELECT A FROM {} ORDER BY A".format(table_name)).fetchall()
            assert results == [[23], [None], [None]]


@for_each_database
def test_multiple_columns(dsn, configuration):
    with open_cursor(configuration) as cursor:
        with query_fixture(cursor, configuration, 'INSERT TWO INTEGER COLUMNS') as table_name:
            columns = [array([17, 23, 42], dtype='int64'), array([3, 2, 1], dtype='int64')]
            cursor.executemanycolumns("INSERT INTO {} VALUES (?, ?)".format(table_name), columns)

            results = cursor.execute("SELECT A, B FROM {} ORDER BY A".format(table_name)).fetchall()
            assert results == [[17, 3], [23, 2], [42, 1]]
