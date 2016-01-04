from __future__ import absolute_import

from .exceptions import translate_exceptions, Error
from .cursor import Cursor


class Connection(object):
    def _assert_valid(self):
        if self.impl is None:
            raise Error("Connection already closed")

    def __init__(self, impl):
        self.impl = impl
        self.cursors = []

    @translate_exceptions
    def cursor(self):
        """Create a cursor object"""
        self._assert_valid()
        c = Cursor(self.impl.cursor())
        self.cursors.append(c)
        return c

    @translate_exceptions
    def commit(self):
        self._assert_valid()
        self.impl.commit()

    def close(self):
        self._assert_valid()
        for c in self.cursors:
            c.close()
        self.cursors = []
        self.impl = None
