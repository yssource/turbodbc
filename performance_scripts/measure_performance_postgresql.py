import pyodbc
import turbodbc
import numpy
import json
import pgdb # PyGreSQL
import psycopg2
from datetime import datetime, date

def connect(api, dsn):
    if api == "pyodbc":
        return pyodbc.connect(dsn=dsn)
    if api == "PyGreSQL":
        return pgdb.connect(database='test_db', host='localhost:5432', user='postgres', password='test')
    if api == "psycopg2":
        return psycopg2.connect("dbname='test_db' user='postgres' host='localhost' password='test'")
    else:
        return turbodbc.connect(dsn, parameter_sets_to_buffer=100000, rows_to_buffer=100000, use_async_io=True)

def _column_data(column_type):
    if column_type == 'INTEGER':
        return 42
    if column_type == 'DOUBLE PRECISION':
        return 3.14
    if column_type == 'DATE':
        return date(2016, 01, 02)
    if 'VARCHAR' in column_type:
        return 'test data'
    if column_type == 'TIMESTAMP':
        return datetime(2016, 01, 02, 03, 04, 05)
    raise RuntimeError("Unknown column type {}".format(column_type))


def prepare_test_data(cursor, column_types, powers_of_two_lines):
    cursor.execute("DROP TABLE IF EXISTS test_performance")
    columns = ['col{} {}'.format(i, type) for i, type in zip(xrange(len(column_types)), column_types)]
    cursor.execute('CREATE TABLE test_performance ({})'.format(', '.join(columns)))

    data = [_column_data(type) for type in column_types]
#     try:
    cursor.execute('INSERT INTO test_performance VALUES ({})'.format(', '.join('?' for _ in columns)), data)
#     except:
#     cursor.execute('INSERT INTO test_performance VALUES ({})'.format(', '.join('%s' for _ in columns)), data)

    for _ in xrange(powers_of_two_lines):
        cursor.execute('INSERT INTO test_performance SELECT * FROM test_performance')


def _fetchallnumpy(cursor):
    cursor.fetchallnumpy()

def _stream_to_ignore(cursor):
    for _ in cursor:
        pass


def _stream_to_list(cursor):
    [row for row in cursor]


def measure(cursor, extraction_method):
    cursor.execute('SELECT * FROM test_performance')
    start = datetime.now()
    extraction_method(cursor)
    stop = datetime.now()
    return (stop - start).total_seconds()


# if __name__ == "__main__":
powers_of_two = 10
n_rows = 2**powers_of_two
n_runs = 10
column_types = ['INTEGER', 'INTEGER', 'DOUBLE PRECISION', 'DOUBLE PRECISION', 'VARCHAR(20)', 'DATE', 'TIMESTAMP']
api = 'turbodbc'
# extraction_method = _stream_to_ignore
# extraction_method = _stream_to_list
extraction_method = _fetchallnumpy
database = 'psql'

connection = connect(api, database)
cursor = connection.cursor()

print "Performing benchmark with {} rows".format(n_rows)
prepare_test_data(cursor, column_types, powers_of_two)

runs = []
for r in xrange(n_runs):
    print "Run #{}".format(r + 1)
    runs.append(measure(cursor, extraction_method))

runs = numpy.array(runs)
results = {'number_of_runs': n_runs,
           'rows_per_run': n_rows,
           'column_types': column_types,
           'api': api,
           'extraction_method': extraction_method.__name__,
           'database': database,
           'timings': {'best': runs.min(),
                       'worst': runs.max(),
                       'mean': runs.mean(),
                       'standard_deviation': runs.std()},
           'rates': {'best': n_rows / runs.min(),
                     'worst': n_rows / runs.max(),
                     'mean': n_rows * numpy.reciprocal(runs).mean(),
                     'standard_deviation': n_rows * numpy.reciprocal(runs).std()}}

print json.dumps(results, indent=4, separators=(',', ': '))
file_name = 'results_{}_{}{}.json'.format(database,
                                          api,
                                          extraction_method.__name__)
with open(file_name, 'w') as file:
    json.dump(results, file, indent=4, separators=(',', ': '))

print "Wrote results to file {}".format(file_name)
