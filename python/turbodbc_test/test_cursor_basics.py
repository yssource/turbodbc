import pytest

from turbodbc import connect, InterfaceError

from helpers import for_one_database


@for_one_database
def test_new_cursor_properties(dsn, configuration):
    connection = connect(dsn)
    cursor = connection.cursor()

    # https://www.python.org/dev/peps/pep-0249/#rowcount
    assert cursor.rowcount == -1
    assert None == cursor.description
    assert cursor.arraysize == 1


@for_one_database
def test_closed_cursor_raises_when_used(dsn, configuration):
    connection = connect(dsn)
    cursor = connection.cursor()

    cursor.close()

    with pytest.raises(InterfaceError):
        cursor.execute("SELECT 42")

    with pytest.raises(InterfaceError):
        cursor.executemany("SELECT 42")

    with pytest.raises(InterfaceError):
        cursor.fetchone()

    with pytest.raises(InterfaceError):
        cursor.fetchmany()

    with pytest.raises(InterfaceError):
        cursor.fetchall()

    with pytest.raises(InterfaceError):
        cursor.next()


@for_one_database
def test_closing_twice_does_not_raise(dsn, configuration):
    connection = connect(dsn)
    cursor = connection.cursor()

    cursor.close()
    cursor.close()


@for_one_database
def test_open_cursor_without_result_set_raises(dsn, configuration):
    connection = connect(dsn)
    cursor = connection.cursor()

    with pytest.raises(InterfaceError):
        cursor.fetchone()


@for_one_database
def test_setinputsizes_does_not_raise(dsn, configuration):
    """
    It is legal for setinputsizes() to do nothing, so anything except
    raising an exception is ok
    """
    cursor = connect(dsn).cursor()
    cursor.setinputsizes([10, 20])


@for_one_database
def test_setoutputsize_does_not_raise(dsn, configuration):
    """
    It is legal for setinputsizes() to do nothing, so anything except
    raising an exception is ok
    """
    cursor = connect(dsn).cursor()
    cursor.setoutputsize(1000, 42) # with column
    cursor.setoutputsize(1000, column=42) # with column
    cursor.setoutputsize(1000) # without column