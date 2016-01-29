from unittest import TestCase

from turbodbc import connect, InterfaceError, DatabaseError


dsn = "Exasol R&D test database"


class TestConnection(TestCase):
    def test_closed_connection_raises_when_used(self):
        connection = connect(dsn)

        connection.close()

        with self.assertRaises(InterfaceError):
            connection.cursor()

        with self.assertRaises(InterfaceError):
            connection.commit()

    def test_closing_twice_is_ok(self):
        connection = connect(dsn)
 
        connection.close()
        connection.close()

    def test_closing_connection_closes_all_cursors(self):
        connection = connect(dsn)
        cursor_1 = connection.cursor()
        cursor_2 = connection.cursor()
        connection.close()

        with self.assertRaises(InterfaceError):
            cursor_1.execute("SELECT 42")

        with self.assertRaises(InterfaceError):
            cursor_2.execute("SELECT 42")

    def test_no_autocommit(self):
        connection = connect(dsn)

        connection.cursor().execute('CREATE TABLE test_no_autocommit (a INTEGER)')
        connection.close()

        connection = connect(dsn)
        with self.assertRaises(DatabaseError):
            connection.cursor().execute('SELECT * FROM test_no_autocommit')

    def test_commit(self):
        connection = connect(dsn)

        connection.cursor().execute('CREATE TABLE test_commit (a INTEGER)')
        connection.commit()

        connection.close()

        connection = connect(dsn)
        cursor = connection.cursor()
        cursor.execute('SELECT * FROM test_commit')
        results = cursor.fetchall()
        self.assertEqual(results, [])
        
        cursor.execute('DROP TABLE test_commit')
        connection.commit()