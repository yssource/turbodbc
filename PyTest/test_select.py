from unittest import TestCase

from contextlib import contextmanager

import pydbc
import json
import random


@contextmanager
def query_fixture(cursor, file_name, fixture_key):
    """
    Context manager used to set up fixtures for setting up queries.
    :param file_name: Name of file which contains fixtures in JSON format
    :param fixture_key: Identifies the fixture
    
    The context manager performs the following tasks:
    * Execute all queries listed in the fixture's "setup" section
    * Return the query listed in the fixture's "payload" section
    * Execute all queries listed in the fixture's "teardown" section
    
    The fixture file should have the following format:
    
    {
        "my_fixture_key": {
            "setup": ["A POTENTIALLY EMPTY LIST OF SQL QUERIES"],
            "payload": "SELECT 42",
            "teardown": ["MAY INCLUDE", "MULTIPLE QUERIES"]
        }
    }
    
    Setup and teardown sections are optional. Queries may contain
    "{table_name}" to be replaced with a random table name.
    """
    with open(file_name, 'r') as file:
        fixture = json.load(file)[fixture_key]

    table_name = table_name = 'test_{}'.format(random.randint(0, 1000000000))
    
    def _execute_queries(section_key):
        queries = fixture.get(section_key, [])
        if not isinstance(queries, list):
            queries = [queries]

        for query in queries:
            try:
                cursor.execute(query.format(table_name=table_name))
            except Exception as error:
                raise type(error)('Error during {} of query fixture "{}": {}'.format(section_key, fixture_key, error))

    _execute_queries("setup")
    try:
        yield fixture['payload'].format(table_name=table_name)
    finally:
        _execute_queries("teardown")


class SelectBaseTestCase(object):
    """
    Children are expected to provide the following attributes:
    
    self.dsn
    self.supports_row_count
    """

    def setUp(self):
        self.connection = pydbc.connect(self.dsn)
        self.cursor = self.connection.cursor()

    def tearDown(self):
        self.cursor.close()
        self.connection.close()

    def _test_single_row_result_set(self, query, expected_row):
        self.cursor.execute(query)

        if self.supports_row_count:
            self.assertEqual(self.cursor.rowcount, 1)
        else:
            self.assertEqual(self.cursor.rowcount, -1)

        row = self.cursor.fetchone()
        self.assertItemsEqual(row, expected_row)

        row = self.cursor.fetchone()
        self.assertIsNone(row)

    def test_single_row_NULL_result(self):
        self._test_single_row_result_set("SELECT NULL", [None])

    def test_single_row_integer_result(self):
        self._test_single_row_result_set("SELECT 42", [42])

    def test_single_row_string_result(self):
        self._test_single_row_result_set("SELECT 'value'", ["value"])

    def test_single_row_unicode_result(self):
        self._test_single_row_result_set(u"SELECT 'value \u2665'", [u"value \u2665"])

    def test_single_row_large_numeric_result_as_string(self):
        self._test_single_row_result_set("SELECT -1234567890123.123456789", ['-1234567890123.123456789'])

    def test_single_row_multiple_integer_result(self):
        self._test_single_row_result_set("SELECT 40, 41, 42, 43", [40, 41, 42, 43])

    def test_single_row_double_result(self):
        with query_fixture(self.cursor, self.schema_file, 'SELECT DOUBLE') as query:
            self.cursor.execute(query)
            row = self.cursor.fetchone()
            self.assertItemsEqual(row, [3.14])

    def test_multiple_row_iterate_result(self):
        self.cursor.execute("delete from test_integer")
        for i in xrange(1,10):
            self.cursor.execute("insert into test_integer values("+str(i)+")")
        self.cursor.execute("select * from test_integer order by a")
        for element in enumerate(self.cursor, start=1):
            self.assertItemsEqual([element[0]], element[1])


# Actual test cases

class TestSelectExasol(SelectBaseTestCase, TestCase):
    dsn = "Exasol R&D test database"
    supports_row_count = True
    schema_file = 'db_scripts_exasol.json'


class TestSelectPostgreSQL(SelectBaseTestCase, TestCase):
    dsn = "PostgreSQL R&D test database"
    supports_row_count = False
    schema_file = 'db_scripts_postgresql.json'


class TestSelectMySQL(SelectBaseTestCase, TestCase):
    dsn = "MySQL R&D test database"
    supports_row_count = True
    schema_file = 'db_scripts_mysql.json'
