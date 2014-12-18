from unittest import TestCase
import pydbc

class TestImportPyDBC(TestCase):
        
    def test_connect(self):
        connection = pydbc.connect("PostgreSQL R&D test database")
        
    def test_execute(self):
        connection = pydbc.connect("PostgreSQL R&D test database")
        cursor = connection.cursor()
        cursor.execute("select 42")
        
    def test_connect_error(self):
        self.assertRaises(pydbc.Error, pydbc.connect, "")
        
if __name__ == '__main__':
    from unittest import main
    main()