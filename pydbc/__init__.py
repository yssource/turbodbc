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
            raise Error("Cursor already closed")
        
    def execute(self, sql):
        """Execute an SQL query"""
        self._assert_valid()
        self.impl.execute(sql)
        self.rowcount = self.impl.get_rowcount()
        
    def fetchone(self):
        result = self.impl.fetchone()
        if len(result) == 0:
            return None 
        else:
            return result  

    def close(self):
        self._assert_valid()
        self.impl = None

    def is_closed(self):
        return self.impl == None

class Connection():
    def _assert_valid(self):
        if self.impl is None:
            raise Error("Connection already closed")
    
    def __init__(self, impl):
        self.impl = impl
        self.cursors = []
        
    def cursor(self):
        """Create a cursor object"""
        self._assert_valid()
        c = cursor(self.impl.cursor())
        self.cursors.append(c)
        return c
    
    def commit(self):
        self._assert_valid()
        self.impl.commit()
        
    def close(self):
        self._assert_valid()
        #TODO: connection needs impl knowledge on cursor. :(
        for c in self.cursors:
            if not c.is_closed():
                c.close()
        self.cursors = []
        self.impl = None


@translate_exceptions
def connect(dsn):
    """Create ODBC connection"""
    return Connection(intern.connect(dsn))
    
