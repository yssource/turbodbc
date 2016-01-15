from unittest import TestCase
import datetime

from turbodbc import Date, Time, Timestamp


class TestConstructors(TestCase):
    def test_date(self):
        d = Date(2016, 1, 4)
        self.assertEqual(d, datetime.date(2016, 1, 4))

    def test_time(self):
        t = Time(1, 2, 3)
        self.assertEqual(t, datetime.time(1, 2, 3))

    def test_timestamp(self):
        ts = Timestamp(2016, 1, 2, 3, 4, 5)
        self.assertEqual(ts, datetime.datetime(2016, 1, 2, 3, 4, 5))