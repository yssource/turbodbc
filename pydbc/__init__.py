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
        
    def execute(self, sql):
        """Execute an SQL query"""
        self.impl.execute(sql)

class connection():
    def __init__(self, impl):
        self.impl = impl
        
    def cursor(self):
        """Create a cursor object"""
        return cursor(self.impl.cursor())
    
    def commit(self):
        self.impl.commit()


@translate_exceptions
def connect(dsn):
    """Create ODBC connection"""
    return connection(intern.connect(dsn))
    