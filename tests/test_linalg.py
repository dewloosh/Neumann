# -*- coding: utf-8 -*-
import unittest
import numpy as np

from neumann.utils import random_pos_semidef_matrix, random_posdef_matrix
from neumann.logical import ispossemidef, isposdef
from neumann.linalg import (ReferenceFrame, Vector, inv3x3, det3x3, 
                            inv2x2u, inv3x3u, inv2x2, det2x2)
from neumann.linalg.solve import solve, reduce, _measure


class TestPosDef(unittest.TestCase):
    
    def test_random_pos_semidef(self, N=2):
        """
        Tests the creation of random, positive semidefinite matrices.
        """
        assert ispossemidef(random_pos_semidef_matrix(N))

    def test_random_posdef(self, N=2):
        """
        Tests the creation of random, positive definite matrices.
        """
        assert isposdef(random_posdef_matrix(N))


class TestTransform(unittest.TestCase):

    def test_tr_vector_1(self, i=1, a=120.):
        """
        Applies a random rotation of a frame around a random axis
        and tests the transformation of components.
        """

        # the original frame
        A = ReferenceFrame(dim=3)

        # the original vector
        vA = Vector([1., 0., 0.], frame=A)

        # random rotation
        amounts = [0., 0., 0.]
        amounts[i] = a * np.pi / 180
        B = A.orient_new('Body', amounts, 'XYZ')

        # Image of vA in B
        vB = Vector(vA.show(B), frame=B)

        # test if the image of vB in A is the same as vA
        assert np.all(np.isclose(vB.show(A), vA.array))

    def test_tr_vector_2(self, angle=22.):
        """
        Tests the equivalence of a series of relative transformations
        against an absolute transformation.
        """
        A = ReferenceFrame(dim=3)
        vA = Vector([1.0, 0., 0.0], frame=A)
        B = A.orient_new('Body', [0., 0., 0], 'XYZ')
        N = 3
        dtheta = angle * np.pi / 180 / N
        theta = 0.
        for _ in range(N):
            B.orient('Body', [0., 0., dtheta], 'XYZ')
            vB_rel = Vector(vA.array, frame=B)
            theta += dtheta
            vB_tot = vA.orient_new('Body', [0., 0., theta], 'XYZ')
            assert np.all(np.isclose(vB_rel.show(), vB_tot.show()))


class TestLinsolve(unittest.TestCase):
    
    def test_linsolve_2x2(self):
        # a random system
        A = random_posdef_matrix(2)
        b = np.random.rand(2)
        # control solution
        x = np.linalg.solve(A, b)
        # test solution #1
        x_ = inv2x2(A) @ b
        diff = np.abs(x - x_)
        err = np.dot(diff, diff)
        assert err < 1e-12
        # test solution #2
        invA = np.zeros_like(A)
        inv2x2u(A, invA)
        x_ = invA @ b
        diff = np.abs(x - x_)
        err = np.dot(diff, diff)
        assert err < 1e-12

    def test_linsolve_3x3(self):
        # a random system
        A = random_posdef_matrix(3)
        b = np.random.rand(3)
        # control solution
        x = np.linalg.solve(A, b)
        # test solution #1
        x_ = inv3x3(A) @ b
        diff = np.abs(x - x_)
        err = np.dot(diff, diff)
        assert err < 1e-12
        # test solution #2
        invA = np.zeros_like(A)
        inv3x3u(A, invA)
        x_ = invA @ b
        diff = np.abs(x - x_)
        err = np.dot(diff, diff)
        assert err < 1e-12

    def test_det_2x2(self):
        A = random_posdef_matrix(2)
        det = np.linalg.det(A)
        det1 = det2x2(A)
        diff1 = np.abs(det - det1)
        assert diff1 < 1e-12
    
    def test_det_3x3(self):
        A = random_posdef_matrix(3)
        det = np.linalg.det(A)
        det1 = det3x3(A)
        diff1 = np.abs(det - det1)
        assert diff1 < 1e-12

    def test_Gauss_Jordan(self):
        A = np.array([[3, 1, 2], [1, 1, 1], [2, 1, 2]], dtype=float)
        B = np.array([11, 6, 10], dtype=float)
        presc_bool = np.full(B.shape, False, dtype=bool)
        presc_val = np.zeros(B.shape, dtype=float)

        x_J = solve(A, B, method='Jordan')
        x_GJ = solve(A, B, method='Gauss-Jordan')
        x_np = np.linalg.solve(A, B)
        x_np_jit = solve(A, B, method='numpy')
        A_GJ, b_GJ = reduce(A, B, method='Gauss-Jordan')
        A_J, b_J = reduce(A, B, method='Jordan')

        B = np.stack([B, B], axis=1)
        X_J = solve(A, B, method='Jordan')
        X_GJ = solve(A, B, method='Gauss-Jordan')

        A = np.array([[1, 2, 1, 1], [2, 5, 2, 1], [1, 2, 3, 2],
                      [1, 1, 2, 2]], dtype=float)
        b = np.array([12, 15, 22, 17], dtype=float)
        ifpre = np.array([0, 1, 0, 0], dtype=int)
        fixed = np.array([0, 2, 0, 0], dtype=float)
        A_GJ, b_GJ = reduce(A, b, ifpre, fixed, 'Gauss-Jordan')
        A_J, b_J = reduce(A, b, ifpre, fixed, 'Jordan')
        X_GJ, r_GJ = solve(A, b, ifpre, fixed, 'Gauss-Jordan')
        X_J, r_J = solve(A, b, ifpre, fixed, 'Jordan')

        A = np.array([[3, 1, 2], [1, 1, 1], [2, 1, 2]], dtype=float)
        b = np.array([11, 6, 10], dtype=float)
        _measure(A, b, 10)


if __name__ == "__main__":

    unittest.main()
