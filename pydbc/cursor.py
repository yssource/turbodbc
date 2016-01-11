from __future__ import absolute_import

from .exceptions import translate_exceptions, InterfaceError


class Cursor(object):
    def __init__(self, impl):
        self.impl = impl
        self.rowcount = -1
        self.description = None
        self._arraysize = 1

    def __iter__(self):
        return self

    def next(self):
        element = self.fetchone()
        if element is None:
            raise StopIteration
        else:
            return element

    def _assert_valid(self):
        if self.impl is None:
            raise InterfaceError("Cursor already closed")

    @property
    def arraysize(self):
        return self._arraysize

    @arraysize.setter
    def arraysize(self, value):
        self._arraysize = value

    @translate_exceptions
    def execute(self, sql, parameters=None):
        """Execute an SQL query"""
        self._assert_valid()
        self.impl.prepare(sql)
        if parameters:
            self.impl.add_parameter_set(parameters)
        self.impl.execute()
        self.rowcount = self.impl.get_row_count()

    @translate_exceptions
    def executemany(self, sql, parameters=None):
        """Execute an SQL query"""
        self._assert_valid()
        self.impl.prepare(sql)

        if parameters:
            for parameter_set in parameters:
                self.impl.add_parameter_set(parameter_set)

        self.impl.execute()
        self.rowcount = self.impl.get_row_count()

    @translate_exceptions
    def fetchone(self):
        self._assert_valid()
        result = self.impl.fetchone()
        if len(result) == 0:
            return None 
        else:
            return result  

    @translate_exceptions    
    def fetchall(self):
        #can be optimized by implementing it in C++. 
        #But has to make sure that really all remaining rows are fetched,
        #thereby finishing and closing the associated internal result set buffer.
        def rows():
            row = self.fetchone()
            while (row):
                yield row
                row = self.fetchone()
        return [row for row in rows()]

    @translate_exceptions    
    def fetchmany(self, size=arraysize):
        def rows(maxrows):
            rowcount = 1
            row = self.fetchone()
            yield row
            while rowcount<size:
                rowcount+=1
                row = self.fetchone()
                if not row:
                    break
                yield row
                
        if (size<=0):
            return []
        return [row for row in rows(size)]

    def close(self):
        self.impl = None
