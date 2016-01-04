from __future__ import absolute_import

from functools import wraps
from exceptions import StandardError

from .pydbc_intern import Error as InternError


class Error(StandardError):
    pass


def translate_exceptions(f):
    @wraps(f)
    def wrapper(*args, **kwds):
        try:
            return f(*args, **kwds)
        except InternError as e:
            raise Error(str(e))
    return wrapper