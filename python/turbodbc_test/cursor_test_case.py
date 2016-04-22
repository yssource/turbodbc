from unittest import TestCase
from contextlib import contextmanager

import json

import turbodbc


TestCase.maxDiff = None


@contextmanager
def open_connection(configuration, parameter_sets_to_buffer=100):
    dsn = configuration['data_source_name']
    connection = turbodbc.connect(dsn, parameter_sets_to_buffer=parameter_sets_to_buffer)
    yield connection
    connection.close()


@contextmanager
def open_cursor(configuration, parameter_sets_to_buffer=100):
    with open_connection(configuration, parameter_sets_to_buffer) as connection:
        cursor = connection.cursor()
        yield cursor
        cursor.close()


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
            cls.dsn = cls.configuration['data_source_name']

    def setUp(self):
        self.parameter_sets_to_buffer = 100
        self.connection = turbodbc.connect(self.dsn,
                                           parameter_sets_to_buffer=self.parameter_sets_to_buffer)
        self.cursor = self.connection.cursor()

    def tearDown(self):
        self.cursor.close()
        self.connection.close()