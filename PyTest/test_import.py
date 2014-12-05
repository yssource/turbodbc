from unittest import TestCase
from pydbc import connect

class TestImportPyDBC(TestCase):
        
    def test_connect(self):
        self.assertEquals(17, connect("PostgreSQL R&D test database"))
        
if __name__ == '__main__':
    from unittest import main
    main()