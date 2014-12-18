import pydbc_intern as intern
from exceptions import StandardError
from functools import wraps

class Error(StandardError):
    pass


def translate_exceptions(f):
    @wraps(f)
    def wrapper(*args, **kwds):
        try:
            return f(*args, **kwds)
        except intern.Error as e:
            raise Error(str(e))
    return wrapper

class cursor():
    def __init__(self, impl):
        self.impl = impl
        self.rowcount = -1
        self.description = None
    
    def _assert_valid(self):
        if self.impl is None:
            raise Error("Cursor already closed")
        
    def execute(self, sql):
        """Execute an SQL query"""
        self._assert_valid()
        self.impl.execute(sql)

    def close(self):
        self._assert_valid()
        self.impl = None

class connection():
    def _assert_valid(self):
        if self.impl is None:
            raise Error("Connection already closed")
    
    def __init__(self, impl):
        self.impl = impl
        
    def cursor(self):
        """Create a cursor object"""
        self._assert_valid()
        return cursor(self.impl.cursor())
    
    def commit(self):
        self._assert_valid()
        self.impl.commit()
        
    def close(self):
        self._assert_valid()
        self.impl = None


@translate_exceptions
def connect(dsn):
    """Create ODBC connection"""
    return connection(intern.connect(dsn))
    
