import pydbc_intern as intern

from functools import wraps
from exceptions import StandardError


class Error(StandardError):
    pass


def translate_exceptions(f):
    @wraps(f)
    def wrapper(*args, **kwds):
        try:
            return f(*args, **kwds)
        except intern.Error as e:
            raise Error(str(e))
    return wrapper