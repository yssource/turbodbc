import json
import pytest


def _get_configuration(file_name):
    with open(file_name, 'r') as f:
        return json.load(f)


def _get_configurations():
    config_files = ['query_fixtures_exasol.json',
                    'query_fixtures_mysql.json',
                    'query_fixtures_postgresql.json']

    configurations = [_get_configuration(file_name) for file_name in config_files]
    return [(conf['data_source_name'], conf) for conf in configurations]


"""
Use this decorator to execute a test function once for each database configuration.

Please note the test function *must* take the parameters `dsn` and `configuration`,
and in that order.

Example:

@for_each_database
def test_important_stuff(dsn, configuration):
    assert 1 == 2
"""
for_each_database = pytest.mark.parametrize("dsn,configuration",
                                            _get_configurations())