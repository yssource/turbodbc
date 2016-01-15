class DataType(object):
    def __init__(self, value):
        self._value = value

    def __eq__(self, other):
        return self._value == other._value

    def __ne__(self, other):
        return not (self == other)

# data types according to https://www.python.org/dev/peps/pep-0249/#type-objects-and-constructors
STRING = DataType('string type')
BINARY = DataType('binary type')
NUMBER = DataType('number type')
DATETIME = DataType('datetime type')
ROWID = DataType('row ID type')