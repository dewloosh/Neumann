import unittest
import doctest

import sympy as sy
import numpy as np

import neumann.function as fnc
from neumann.function import Function
from neumann.function import Equality, InEquality, Relation
from neumann.approx.lagrange import gen_Lagrange_1d


def load_tests(loader, tests, ignore):  # pragma: no cover
    tests.addTests(doctest.DocTestSuite(fnc.relation))
    return tests


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

    def test_Relation(self):
        variables = ['x1', 'x2', 'x3', 'x4'] 
        x1, _, x3, x4 = syms = sy.symbols(variables, positive=True) 
        r = Relation(x1 + 2*x3 + x4 - 4, variables=syms)
        r.operator
        r = Relation(x1 + 2*x3 + x4 - 4, variables=syms, op=lambda x, y: x <= y)
    
    def test_Equality(self):
        variables = ['x1', 'x2', 'x3', 'x4'] 
        x1, _, x3, x4 = syms = sy.symbols(variables, positive=True) 
        eq1 = Equality(x1 + 2*x3 + x4 - 4, variables=syms)
        eq1.to_eq() 
        eq1.operator
    
    def test_InEquality(self):
        gt = InEquality('x + y', op='>')
        assert not gt.relate([0.0, 0.0])

        ge = InEquality('x + y', op='>=')
        assert ge.relate([0.0, 0.0])

        le = InEquality('x + y', op=lambda x, y: x <= y)
        assert le.relate([0.0, 0.0])

        lt = InEquality('x + y', op=lambda x, y: x < y)
        assert not lt.relate([0.0, 0.0])
        
        failed_properly = False
        try:
            InEquality('x + y')
        except ValueError:
            failed_properly = True
        finally:
            self.assertTrue(failed_properly)


if __name__ == "__main__":

    unittest.main()
