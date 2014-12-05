from unittest import TestCase

class TestImportPyDBC(TestCase):
    def test_import(self):
        import pydbc
        
if __name__ == '__main__':
    from unittest import main
    main()