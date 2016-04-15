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

        if self.capabilities['supports_row_count']:
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
        with query_fixture(self.cursor, self.configuration, 'SELECT DOUBLE') as query:
            self.cursor.execute(query)
            row = self.cursor.fetchone()
            self.assertItemsEqual(row, [3.14])

    def test_single_row_date_result(self):
        self._test_single_row_result_set("SELECT CAST('2015-12-31' AS DATE) AS a",
                                         [datetime.date(2015, 12, 31)])

    def test_single_row_timestamp_result(self):
        with query_fixture(self.cursor, self.configuration, 'SELECT TIMESTAMP') as query:
            self.cursor.execute(query)
            row = self.cursor.fetchone()
            self.assertItemsEqual(row, [datetime.datetime(2015, 12, 31, 1, 2, 3)])

    def test_single_row_large_numeric_result_as_string(self):
        self._test_single_row_result_set("SELECT -1234567890123.123456789", ['-1234567890123.123456789'])

    def test_single_row_multiple_columns(self):
        self._test_single_row_result_set("SELECT 40, 41, 42, 43", [40, 41, 42, 43])

    def test_fetchone(self):
        with query_fixture(self.cursor, self.configuration, 'SELECT MULTIPLE INTEGERS') as query:
            self.cursor.execute(query)
            row = self.cursor.fetchone()
            self.assertItemsEqual(row, [42])
            row = self.cursor.fetchone()
            self.assertItemsEqual(row, [43])
            row = self.cursor.fetchone()
            self.assertItemsEqual(row, [44])

            row = self.cursor.fetchone()
            self.assertIsNone(row)

    def test_fetchall(self):
        with query_fixture(self.cursor, self.configuration, 'SELECT MULTIPLE INTEGERS') as query:
            self.cursor.execute(query)
            rows = self.cursor.fetchall()
            self.assertEqual(len(rows), 3)
            self.assertItemsEqual(rows[0], [42])
            self.assertItemsEqual(rows[1], [43])
            self.assertItemsEqual(rows[2], [44])

    def test_fetchmany_with_default_arraysize(self):
        with query_fixture(self.cursor, self.configuration, 'SELECT MULTIPLE INTEGERS') as query:
            self.cursor.execute(query)
            rows = self.cursor.fetchmany()
            self.assertEqual(len(rows), 1)
            self.assertItemsEqual(rows[0], [42])

    def test_fetchmany_with_arraysize_parameter(self):
        with query_fixture(self.cursor, self.configuration, 'SELECT MULTIPLE INTEGERS') as query:
            self.cursor.execute(query)
            arraysize_parameter = 2

            rows = self.cursor.fetchmany(arraysize_parameter)
            self.assertEqual(len(rows), arraysize_parameter)
            self.assertItemsEqual(rows[0], [42])
            self.assertItemsEqual(rows[1], [43])

            # arraysize exceeds number of remaining rows
            rows = self.cursor.fetchmany(arraysize_parameter)
            self.assertEqual(len(rows), 1)
            self.assertItemsEqual(rows[0], [44])

    def test_fetchmany_with_global_arraysize(self):
        with query_fixture(self.cursor, self.configuration, 'SELECT MULTIPLE INTEGERS') as query:
            self.cursor.execute(query)

            arraysize_parameter = 2
            self.cursor.arraysize = arraysize_parameter

            rows = self.cursor.fetchmany()
            self.assertEqual(len(rows), arraysize_parameter)
            self.assertItemsEqual(rows[0], [42])
            self.assertItemsEqual(rows[1], [43])

            # arraysize exceeds number of remaining rows
            rows = self.cursor.fetchmany()
            self.assertEqual(len(rows), 1)
            self.assertItemsEqual(rows[0], [44])

    def test_fetchmany_with_bad_arraysize_parameter_raises(self):
        with query_fixture(self.cursor, self.configuration, 'SELECT MULTIPLE INTEGERS') as query:
            self.cursor.execute(query)

            with self.assertRaises(turbodbc.InterfaceError):
                self.cursor.fetchmany(-1)
            with self.assertRaises(turbodbc.InterfaceError):
                self.cursor.fetchmany(0)

    def test_fetchmany_with_bad_global_arraysize_raises(self):
        with query_fixture(self.cursor, self.configuration, 'SELECT MULTIPLE INTEGERS') as query:
            self.cursor.execute(query)

            self.cursor.arraysize = -1
            with self.assertRaises(turbodbc.InterfaceError):
                self.cursor.fetchmany()

            self.cursor.arraysize = 0
            with self.assertRaises(turbodbc.InterfaceError):
                self.cursor.fetchmany()

    def test_number_of_rows_exceeds_buffer_size(self):
        with query_fixture(self.cursor, self.configuration, 'INSERT INTEGER') as table_name:
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
            if self.capabilities['reports_column_names_as_upper_case']:
                return string.upper()
            else:
                return string

        with query_fixture(self.cursor, self.configuration, 'DESCRIPTION') as table_name:
            self.cursor.execute("SELECT * FROM {}".format(table_name))

            nullness_for_null_column = not self.capabilities['indicates_null_columns']

            expected = [(fix_case('as_int'), turbodbc.NUMBER, None, None, None, None, True),
                        (fix_case('as_double'), turbodbc.NUMBER, None, None, None, None, True),
                        (fix_case('as_varchar'), turbodbc.STRING, None, None, None, None, True),
                        (fix_case('as_date'), turbodbc.DATETIME, None, None, None, None, True),
                        (fix_case('as_timestamp'), turbodbc.DATETIME, None, None, None, None, True),
                        (fix_case('as_int_not_null'), turbodbc.NUMBER, None, None, None, None, nullness_for_null_column)]
            self.assertEqual(expected, self.cursor.description)

# Actual test cases

class TestCursorSelectExasol(SelectTests, CursorTestCase):
    fixture_file_name = 'query_fixtures_exasol.json'


class TestCursorSelectPostgreSQL(SelectTests, CursorTestCase):
    fixture_file_name = 'query_fixtures_postgresql.json'


class TestCursorSelectMySQL(SelectTests, CursorTestCase):
    fixture_file_name = 'query_fixtures_mysql.json'
