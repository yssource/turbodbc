from contextlib import contextmanager

import json
import random


@contextmanager
def query_fixture(cursor, file_name, fixture_key):
    """
    Context manager used to set up fixtures for setting up queries.
    :param file_name: Name of file which contains fixtures in JSON format
    :param fixture_key: Identifies the fixture
    
    The context manager performs the following tasks:
    * Execute all queries listed in the fixture's "setup" section
    * Return the query listed in the fixture's "payload" section
    * Execute all queries listed in the fixture's "teardown" section
    
    The fixture file should have the following format:
    
    {
        "my_fixture_key": {
            "setup": ["A POTENTIALLY EMPTY LIST OF SQL QUERIES"],
            "payload": "SELECT 42",
            "teardown": ["MAY INCLUDE", "MULTIPLE QUERIES"]
        }
    }
    
    Setup and teardown sections are optional. Queries may contain
    "{table_name}" to be replaced with a random table name.
    """
    with open(file_name, 'r') as file:
        fixture = json.load(file)[fixture_key]

    table_name = table_name = 'test_{}'.format(random.randint(0, 1000000000))
    
    def _execute_queries(section_key):
        queries = fixture.get(section_key, [])
        if not isinstance(queries, list):
            queries = [queries]

        for query in queries:
            try:
                cursor.execute(query.format(table_name=table_name))
            except Exception as error:
                raise type(error)('Error during {} of query fixture "{}": {}'.format(section_key, fixture_key, error))

    _execute_queries("setup")
    try:
        yield fixture['payload'].format(table_name=table_name)
    finally:
        _execute_queries("teardown")
