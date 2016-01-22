import datetime

import turbodbc

from query_fixture import query_fixture
from cursor_test_case import CursorTestCase


class SelectTests(object):
    """
    Parent class for database-specific SELECT tests. Children are expected to provide
    the following attributes:

    self.supports_row_count
    self.indicates_null_columns
    self.reports_column_names_as_upper_case
    """
    def test_too_many_parameters_raise(self):
        with self.assertRaises(turbodbc.Error):
            self.cursor.execute("SELECT 42", [42])

        with self.assertRaises(turbodbc.Error):
            self.cursor.executemany("SELECT 42", [[42]])

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

    def test_single_row_bool_result(self):
        self._test_single_row_result_set("SELECT True", [True])
        self._test_single_row_result_set("SELECT False", [False])

    def test_single_row_string_result(self):
        self._test_single_row_result_set("SELECT 'value'", ["value"])

    def test_single_row_unicode_result(self):
        self._test_single_row_result_set(u"SELECT 'value \u2665'", [u"value \u2665"])

    def test_single_row_double_result(self):
        with query_fixture(self.cursor, self.fixtures, 'SELECT DOUBLE') as query:
            self.cursor.execute(query)
            row = self.cursor.fetchone()
            self.assertItemsEqual(row, [3.14])

    def test_single_row_date_result(self):
        self._test_single_row_result_set("SELECT CAST('2015-12-31' AS DATE) AS a",
                                         [datetime.date(2015, 12, 31)])

    def test_single_row_timestamp_result(self):
        with query_fixture(self.cursor, self.fixtures, 'SELECT TIMESTAMP') as query:
            self.cursor.execute(query)
            row = self.cursor.fetchone()
            self.assertItemsEqual(row, [datetime.datetime(2015, 12, 31, 1, 2, 3)])

    def test_single_row_large_numeric_result_as_string(self):
        self._test_single_row_result_set("SELECT -1234567890123.123456789", ['-1234567890123.123456789'])

    def test_single_row_multiple_columns(self):
        self._test_single_row_result_set("SELECT 40, 41, 42, 43", [40, 41, 42, 43])

    def test_multiple_rows(self):
        with query_fixture(self.cursor, self.fixtures, 'SELECT MULTIPLE INTEGERS') as query:
            self.cursor.execute(query)
            row = self.cursor.fetchone()
            self.assertItemsEqual(row, [42])
            row = self.cursor.fetchone()
            self.assertItemsEqual(row, [43])
            row = self.cursor.fetchone()
            self.assertItemsEqual(row, [44])

            row = self.cursor.fetchone()
            self.assertIsNone(row)

    def test_number_of_rows_exceeds_buffer_size(self):
        with query_fixture(self.cursor, self.fixtures, 'INSERT INTEGER') as table_name:
            numbers = 123
            for i in xrange(numbers):
                self.cursor.execute("INSERT INTO {} VALUES ({})".format(table_name, i))

            self.cursor.execute("SELECT a FROM {}".format(table_name))
            retrieved = self.cursor.fetchall()
            actual_sum = sum([row[0] for row in retrieved])
            expected_sum = sum(xrange(numbers))
            self.assertEqual(expected_sum, actual_sum)

    def test_description(self):
        self.assertIsNone(self.cursor.description)
        
        def fix_case(string):
            if self.reports_column_names_as_upper_case:
                return string.upper()
            else:
                return string

        with query_fixture(self.cursor, self.fixtures, 'DESCRIPTION') as table_name:
            self.cursor.execute("SELECT * FROM {}".format(table_name))

            nullness_for_null_column = not self.indicates_null_columns

            expected = [(fix_case('as_int'), turbodbc.NUMBER, None, None, None, None, True),
                        (fix_case('as_double'), turbodbc.NUMBER, None, None, None, None, True),
                        (fix_case('as_varchar'), turbodbc.STRING, None, None, None, None, True),
                        (fix_case('as_date'), turbodbc.DATETIME, None, None, None, None, True),
                        (fix_case('as_timestamp'), turbodbc.DATETIME, None, None, None, None, True),
                        (fix_case('as_int_not_null'), turbodbc.NUMBER, None, None, None, None, nullness_for_null_column)]
            self.assertEqual(expected, self.cursor.description)

# Actual test cases

class TestCursorSelectExasol(SelectTests, CursorTestCase):
    dsn = "Exasol R&D test database"
    fixture_file_name = 'query_fixtures_exasol.json'
    supports_row_count = True
    indicates_null_columns = False
    reports_column_names_as_upper_case = True


class TestCursorSelectPostgreSQL(SelectTests, CursorTestCase):
    dsn = "PostgreSQL R&D test database"
    fixture_file_name = 'query_fixtures_postgresql.json'
    supports_row_count = False
    indicates_null_columns = True
    reports_column_names_as_upper_case = False

class TestCursorSelectMySQL(SelectTests, CursorTestCase):
    dsn = "MySQL R&D test database"
    fixture_file_name = 'query_fixtures_mysql.json'
    supports_row_count = True
    indicates_null_columns = True
    reports_column_names_as_upper_case = False
