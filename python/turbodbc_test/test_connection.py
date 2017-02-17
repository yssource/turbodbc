import pytest

from turbodbc import connect, InterfaceError, DatabaseError

from helpers import for_one_database, get_credentials


@for_one_database
def test_cursor_on_closed_connection_raises(dsn, configuration):
    connection = connect(dsn, **get_credentials(configuration))
    connection.close()

    with pytest.raises(InterfaceError):
        connection.cursor()


@for_one_database
def test_closing_twice_is_ok(dsn, configuration):
    connection = connect(dsn, **get_credentials(configuration))

    connection.close()
    connection.close()


@for_one_database
def test_closing_connection_closes_all_cursors(dsn, configuration):
    connection = connect(dsn, **get_credentials(configuration))
    cursor_1 = connection.cursor()
    cursor_2 = connection.cursor()
    connection.close()

    with pytest.raises(InterfaceError):
        cursor_1.execute("SELECT 42")

    with pytest.raises(InterfaceError):
        cursor_2.execute("SELECT 42")


@for_one_database
def test_no_autocommit(dsn, configuration):
    connection = connect(dsn, **get_credentials(configuration))

    connection.cursor().execute('CREATE TABLE test_no_autocommit (a INTEGER)')
    connection.close()

    connection = connect(dsn, **get_credentials(configuration))
    with pytest.raises(DatabaseError):
        connection.cursor().execute('SELECT * FROM test_no_autocommit')


@for_one_database
def test_commit_on_closed_connection_raises(dsn, configuration):
    connection = connect(dsn, **get_credentials(configuration))
    connection.close()

    with pytest.raises(InterfaceError):
        connection.commit()


@for_one_database
def test_commit(dsn, configuration):
    connection = connect(dsn, **get_credentials(configuration))

    connection.cursor().execute('CREATE TABLE test_commit (a INTEGER)')
    connection.commit()

    connection.close()

    connection = connect(dsn, **get_credentials(configuration))
    cursor = connection.cursor()
    cursor.execute('SELECT * FROM test_commit')
    results = cursor.fetchall()
    assert results == []
    
    cursor.execute('DROP TABLE test_commit')
    connection.commit()


@for_one_database
def test_rollback_on_closed_connection_raises(dsn, configuration):
    connection = connect(dsn, **get_credentials(configuration))
    connection.close()

    with pytest.raises(InterfaceError):
        connection.rollback()


@for_one_database
def test_rollback(dsn, configuration):
    connection = connect(dsn, **get_credentials(configuration))

    connection.cursor().execute('CREATE TABLE test_rollback (a INTEGER)')
    connection.rollback()

    with pytest.raises(DatabaseError):
        connection.cursor().execute('SELECT * FROM test_rollback')
