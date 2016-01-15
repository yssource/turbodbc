from __future__ import absolute_import

from .pydbc_intern import connect as intern_connect

from .exceptions import translate_exceptions
from .connection import Connection


@translate_exceptions
def connect(dsn, rows_to_buffer=None, parameter_sets_to_buffer=None):
    """
    Create a connection with the database identified by the dsn
    :param dsn: data source name as given in the odbc.ini file
    :param rows_to_buffer: Number of rows which shall be buffered
                           when result sets are fetched from the server.
                           None means that the default value is used.
    :param parameter_sets_to_buffer: Number of parameter sets (rows) which
                           shall be buffered when bulk queries are sent
                           to the server.
                           None means that the default value is used.
    """
    connection = Connection(intern_connect(dsn))

    if rows_to_buffer:
        connection.impl.rows_to_buffer = rows_to_buffer

    if parameter_sets_to_buffer:
        connection.impl.parameter_sets_to_buffer = parameter_sets_to_buffer

    return connection