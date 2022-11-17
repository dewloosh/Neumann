# -*- coding: utf-8 -*-
import unittest
import sympy as sy
from sympy import symbols
import numpy as np

from neumann.function import Function
from neumann.function import Equality, InEquality
from neumann.approx.lagrange import gen_Lagrange_1d


class TestFunction(unittest.TestCase):

    def test_linearity(self):
        def f0(x=None, y=None):
            return x**2 + y

        def f1(x=None, y=None):
            return np.array([2*x, 1])
        
        f = Function(f0, f1, d=2)
        assert f.linear
        
    def test_sym(self):
        f = gen_Lagrange_1d(N=2)
        f1 = Function(f[1][0], f[1][1], f[1][2])
        f2 = Function(f[2][0], f[2][1], f[2][2])
        assert (f1.linear and f2.linear)
        assert f1.dimension == 1
        assert f2.dimension == 1
        assert np.isclose(f1([-1]), 1.0)
        assert np.isclose(f1([1]), 0.0)
        assert np.isclose(f2([-1]), 0.0)
        assert np.isclose(f2([1]), 1.0)
        f1.coefficients()
        f1.to_latex()
        f1.f([-1]), f1.g([-1]), f1.G([-1])


class TestRelations(unittest.TestCase):

    def test_InEquality(self):
        gt = InEquality('x + y', op='>')
        assert not gt.relate([0.0, 0.0])

        ge = InEquality('x + y', op='>=')
        assert ge.relate([0.0, 0.0])

        le = InEquality('x + y', op=lambda x, y: x <= y)
        assert le.relate([0.0, 0.0])

        lt = InEquality('x + y', op=lambda x, y: x < y)
        assert not lt.relate([0.0, 0.0])


if __name__ == "__main__":

    unittest.main()
