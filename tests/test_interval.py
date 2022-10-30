# -*- coding: utf-8 -*-
import unittest
import numpy as np
from numba import njit

from neumann.interval import Interval


class TestInterval(unittest.TestCase):

    def test_interval(self, lo=1.0, hi=3.0):
        
        @njit()
        def inside_interval(interval, x):
            return interval.lo <= x < interval.hi


        @njit()
        def interval_width(interval):
            return interval.width


        @njit()
        def interval_data(interval):
            return interval.data


        @njit()
        def interval_getitem(interval, i):
            return interval[i]


        @njit()
        def new_interval(lo, hi, data):
            return Interval(lo, hi, data)
        
        data = np.array([1.1, 3.1, 2.1])
        new_interval(lo, hi, data)._arr
        interval_data(new_interval(lo, hi, data))
        interval_getitem(new_interval(lo, hi, data), 0)


if __name__ == "__main__":

    # small example to compile functions
    """lo = 1.0
    hi = 3.0
    data = np.array([1.1, 3.1, 2.1])
    new_interval(lo, hi, data)._arr
    interval_data(new_interval(lo, hi, data))
    interval_getitem(new_interval(lo, hi, data), 0)"""

    #unittest.main()
