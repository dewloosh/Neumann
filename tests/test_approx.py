# -*- coding: utf-8 -*-
import unittest
from neumann.approx.lagrange import gen_Lagrange_1d


class TestLagrange(unittest.TestCase):

    def test_1d(self):
        gen_Lagrange_1d(x=[-1, 0, 1])
        gen_Lagrange_1d(i=[1, 2], sym=True)
        gen_Lagrange_1d(i=[1, 2, 3], sym=False)
        gen_Lagrange_1d(N=3)

if __name__ == "__main__":

    unittest.main()
