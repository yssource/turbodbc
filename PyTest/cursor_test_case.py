from unittest import TestCase

import json

import pydbc


class CursorTestCase(TestCase):
    """
    Children are expected to provide the following attributes:
    
    self.dsn
    self.supports_row_count
    self.fixture_file_name
    """

    @classmethod
    def setUpClass(cls):
        with open(cls.fixture_file_name, 'r') as f:
            cls.fixtures = json.load(f)

    def setUp(self):
        self.connection = pydbc.connect(self.dsn)
        self.cursor = self.connection.cursor()

    def tearDown(self):
        self.cursor.close()
        self.connection.close()