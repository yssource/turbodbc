from unittest import TestCase
from pydbc import connect

class TestImportPyDBC(TestCase):
        
    def test_connect(self):
        connection = connect("PostgreSQL R&D test database")
        
    def test_execute(self):
        connection = connect("PostgreSQL R&D test database")
        cursor = connection.cursor()
        cursor.execute("select 42")
        
if __name__ == '__main__':
    from unittest import main
    main()