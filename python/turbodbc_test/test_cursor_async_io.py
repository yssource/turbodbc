import pytest
import six

from turbodbc import connect

from query_fixture import query_fixture
from helpers import for_one_database, open_cursor


@for_one_database
def test_many_batches_with_async_io(dsn, configuration):
    with open_cursor(configuration, use_async_io=True) as cursor:
        with query_fixture(cursor, configuration, 'INSERT INTEGER') as table_name:
            # insert 2^16 rows
            cursor.execute("INSERT INTO {} VALUES (1)".format(table_name))
            for _ in six.moves.range(16):
                cursor.execute("INSERT INTO {} SELECT * FROM {}".format(table_name,
                                                                        table_name))

            cursor.execute("SELECT * FROM {}".format(table_name))
            assert sum(1 for _ in cursor) == 2**16
