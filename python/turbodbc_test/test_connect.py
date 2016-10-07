import pytest

from turbodbc import connect, DatabaseError
from turbodbc.connect import _make_connection_string
from turbodbc.connection import Connection
from turbodbc_intern import Rows, Megabytes

from helpers import for_one_database


def test_make_connection_string_with_dsn():
    connection_string = _make_connection_string('my_dsn', user='my_user')
    assert connection_string == 'dsn=my_dsn;user=my_user'

def test_make_connection_string_without_dsn():
    connection_string = _make_connection_string(None, user='my_user')
    assert connection_string == 'user=my_user'

@for_one_database
def test_connect_returns_connection_when_successful(dsn, configuration):
    connection = connect(dsn)
    assert isinstance(connection, Connection)

@for_one_database
def test_connect_returns_connection_with_explicit_dsn(dsn, configuration):
    connection = connect(dsn=dsn)
    assert isinstance(connection, Connection)

def test_connect_raises_on_invalid_dsn():
    invalid_dsn = 'This data source does not exist'
    with pytest.raises(DatabaseError):
        connect(invalid_dsn)

@for_one_database
def test_connect_raises_on_invalid_additional_option(dsn, configuration):
    additional_option = {configuration['capabilities']['connection_user_option']: 'invalid user'}
    with pytest.raises(DatabaseError):
        connect(dsn=dsn, **additional_option)

@for_one_database
def test_connect_performance_settings(dsn, configuration):
    connection = connect(dsn=dsn,
                         rows_to_buffer=317,
                         parameter_sets_to_buffer=123,
                         use_async_io=True)

    assert connection.impl.get_buffer_size().rows_to_buffer == 317
    assert connection.impl.parameter_sets_to_buffer == 123
    assert connection.impl.use_async_io == True

@for_one_database
def test_connect_with_rows(dsn, configuration):
    connection = connect(dsn=dsn, read_buffer_size=Rows(317))
    assert connection.impl.get_buffer_size().rows_to_buffer == 317

@for_one_database
def test_connect_with_megabytes(dsn, configuration):
    connection = connect(dsn=dsn, read_buffer_size=Megabytes(1))
    assert connection.impl.get_buffer_size().megabytes_to_buffer == 1