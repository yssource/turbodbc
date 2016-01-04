from __future__ import absolute_import

from .pydbc_intern import connect as intern_connect

from .exceptions import translate_exceptions
from .connection import Connection


@translate_exceptions
def connect(dsn):
    """Create ODBC connection"""
    return Connection(intern_connect(dsn))