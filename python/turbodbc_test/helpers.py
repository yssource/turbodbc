from contextlib import contextmanager

import json
import os
import pytest

import turbodbc

def _get_config_files():
    variable = 'TURBODBC_TEST_CONFIGURATION_FILES'
    try:
        raw = os.environ[variable]
        file_names = raw.split(',')
        return [file_name.strip() for file_name in file_names]
    except KeyError:
        raise KeyError('Please set the environment variable {} to specify the configuration files as a comma-separated list'.format(variable))


def _load_configuration(file_name):
    with open(file_name, 'r') as f:
        return json.load(f)


def _get_configuration(file_name):
    conf = _load_configuration(file_name)
    return (conf['data_source_name'], conf)


def _get_configurations():
    return [_get_configuration(file_name) for file_name in _get_config_files()]


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



"""
Use this decorator to execute a test function once for a single database configuration.

Please note the test function *must* take the parameters `dsn` and `configuration`,
and in that order.

Example:

@for_one_database
def test_important_stuff(dsn, configuration):
    assert 1 == 2
"""
for_one_database = pytest.mark.parametrize("dsn,configuration",
                                           [_get_configuration(_get_config_files()[0])])



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