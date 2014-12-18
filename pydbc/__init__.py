import pydbc_intern as intern


class cursor():
    def __init__(self, impl):
        self.impl = impl
        
    def execute(self, sql):
        """Execute an SQL query"""
        self.impl.execute(sql)
        
    def fetchone(self):
        return self.impl.fetchone() 

class connection():
    def __init__(self, impl):
        self.impl = impl
        
    def cursor(self):
        """Create a cursor object"""
        return cursor(self.impl.cursor())
    
    def commit(self):
        self.impl.commit()


def connect(dsn):
    """Create ODBC connection"""
    return connection(intern.connect(dsn))