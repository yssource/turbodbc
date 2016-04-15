from unittest import TestCase

import json

import turbodbc


TestCase.maxDiff = None

class CursorTestCase(TestCase):
    """
    Children are expected to provide the following attributes:
    
    self.dsn
    self.fixture_file_name
    """

    @classmethod
    def setUpClass(cls):
        with open(cls.fixture_file_name, 'r') as f:
            cls.configuration = json.load(f)
            cls.capabilities = cls.configuration['capabilities']

    def setUp(self):
        self.parameter_sets_to_buffer = 100
        self.connection = turbodbc.connect(self.dsn,
                                           parameter_sets_to_buffer=self.parameter_sets_to_buffer)
        self.cursor = self.connection.cursor()

    def tearDown(self):
        self.cursor.close()
        self.connection.close()