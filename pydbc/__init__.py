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
            raise Error("Cursor already closed")
    
    @property
    def arraysize(self):
        return self._arraysize
    
    @arraysize.setter
    def arraysize(self, value):
        self._arraysize = value
    
    @translate_exceptions
    def execute(self, sql):
        """Execute an SQL query"""
        self._assert_valid()
        self.impl.execute(sql)
        self.rowcount = self.impl.get_rowcount()
        
    @translate_exceptions
    def fetchone(self):
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
        self._assert_valid()
        self.impl = None

    def is_closed(self):
        return self.impl is None

class Connection():
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
        c = cursor(self.impl.cursor())
        self.cursors.append(c)
        return c
    
    @translate_exceptions
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
    
