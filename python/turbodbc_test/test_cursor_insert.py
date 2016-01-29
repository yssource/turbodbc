import datetime

from cursor_test_case import CursorTestCase
from query_fixture import query_fixture


def generate_microseconds_with_precision(digits):
    microseconds = 0;
    for i in xrange(digits):
        microseconds = 10 * microseconds + i + 1
    for i in xrange(6 - digits):
        microseconds *= 10

    return microseconds


class InsertTests(object):
    """
    Parent class for database-specific INSERT tests. Children are expected to provide
    the following attributes:

    self.fractional_second_digits
    """
    def _test_insert_many(self, fixture_name, data):
        with query_fixture(self.cursor, self.fixtures, fixture_name) as table_name:
            self.cursor.executemany("INSERT INTO {} VALUES (?)".format(table_name), data)
            self.assertEqual(len(data), self.cursor.rowcount)
            self.cursor.execute("SELECT a FROM {} ORDER BY a".format(table_name))
            inserted = [list(row) for row in self.cursor.fetchall()]
            self.assertItemsEqual(data, inserted)

    def test_insert_single(self):
        to_insert = [1]

        with query_fixture(self.cursor, self.fixtures, 'INSERT INTEGER') as table_name:
            self.cursor.execute("INSERT INTO {} VALUES (?)".format(table_name), to_insert)
            self.assertEqual(1, self.cursor.rowcount)
            self.cursor.execute("SELECT a FROM {}".format(table_name))
            inserted = [list(row) for row in self.cursor.fetchall()]
            self.assertItemsEqual([to_insert], inserted)

    def test_string_column(self):
        self._test_insert_many('INSERT STRING',
                               [['hello'], ['my'], ['test case']])

    def test_bool_column(self):
        self._test_insert_many('INSERT BOOL',
                               [[True], [True], [False]])

    def test_integer_column(self):
        self._test_insert_many('INSERT INTEGER',
                               [[1], [2], [3]])

    def test_double_column(self):
        self._test_insert_many('INSERT DOUBLE',
                               [[1.23], [2.71], [3.14]])

    def test_date_column(self):
        self._test_insert_many('INSERT DATE',
                               [[datetime.date(2015, 12, 31)],
                                [datetime.date(2016, 1, 15)],
                                [datetime.date(2016, 2, 3)]])

    def test_timestamp_column(self):
        fractional = generate_microseconds_with_precision(self.fractional_second_digits)

        self._test_insert_many('INSERT TIMESTAMP',
                               [[datetime.datetime(2015, 12, 31, 1, 2, 3, fractional)],
                                [datetime.datetime(2016, 1, 15, 4, 5, 6, fractional * 2)],
                                [datetime.datetime(2016, 2, 3, 7, 8, 9, fractional * 3)]])

    def test_null(self):
        self._test_insert_many('INSERT INTEGER',
                               [[None]])

    def test_mixed_data_columns(self):
        # second column has mixed data types in the same column
        # first column makes sure values of "good" columns are not affected
        to_insert = [[23, 1.23],
                     [42, 2]]

        with query_fixture(self.cursor, self.fixtures, 'INSERT MIXED') as table_name:
            self.cursor.executemany("INSERT INTO {} VALUES (?, ?)".format(table_name), to_insert)
            self.assertEqual(len(to_insert), self.cursor.rowcount)
            self.cursor.execute("SELECT a, b FROM {} ORDER BY a".format(table_name))
            inserted = [list(row) for row in self.cursor.fetchall()]
            self.assertItemsEqual(to_insert, inserted)

    def test_no_parameter_list(self):
        with query_fixture(self.cursor, self.fixtures, 'INSERT INTEGER') as table_name:
            self.cursor.executemany("INSERT INTO {} VALUES (?)".format(table_name))
            self.assertEqual(0, self.cursor.rowcount)
            self.cursor.execute("SELECT a FROM {}".format(table_name))
            inserted = [list(row) for row in self.cursor.fetchall()]
            self.assertEqual(0, len(inserted))

    def test_empty_parameter_list(self):
        to_insert = []

        with query_fixture(self.cursor, self.fixtures, 'INSERT INTEGER') as table_name:
            self.cursor.executemany("INSERT INTO {} VALUES (?)".format(table_name), to_insert)
            self.assertEqual(0, self.cursor.rowcount)
            self.cursor.execute("SELECT a FROM {}".format(table_name))
            inserted = [list(row) for row in self.cursor.fetchall()]
            self.assertItemsEqual(to_insert, inserted)

    def test_number_of_rows_exceeds_buffer_size(self):
        numbers = self.parameter_sets_to_buffer * 2 + 17
        self._test_insert_many('INSERT INTEGER',
                               [[i] for i in xrange(numbers)])

    def test_description_after_insert(self):
        with query_fixture(self.cursor, self.fixtures, 'INSERT INTEGER') as table_name:
            self.cursor.execute("INSERT INTO {} VALUES (42)".format(table_name))
            self.assertIsNone(self.cursor.description)


# Actual test cases

class TestCursorInsertExasol(InsertTests, CursorTestCase):
    dsn = "Exasol R&D test database"
    fixture_file_name = 'query_fixtures_exasol.json'
    fractional_second_digits = 3


class TestCursorInsertPostgreSQL(InsertTests, CursorTestCase):
    dsn = "PostgreSQL R&D test database"
    fixture_file_name = 'query_fixtures_postgresql.json'
    fractional_second_digits = 6


class TestCursorInsertMySQL(InsertTests, CursorTestCase):
    dsn = "MySQL R&D test database"
    fixture_file_name = 'query_fixtures_mysql.json'
    fractional_second_digits = 0