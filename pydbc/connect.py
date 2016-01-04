from exception_types import translate_exceptions
from connection import Connection

import pydbc_intern as intern


@translate_exceptions
def connect(dsn):
    """Create ODBC connection"""
    return Connection(intern.connect(dsn))