import unittest

import numpy as np
from numpy import vstack
from scipy.sparse import csr_matrix as csr_scipy
from numba import njit

from neumann.utils import random_pos_semidef_matrix, random_posdef_matrix
from neumann.logical import ispossemidef, isposdef
from neumann.linalg import (ReferenceFrame, RectangularFrame, CartesianFrame, 
                            Vector, inv3x3, det3x3, inv2x2u, inv3x3u, inv2x2, det2x2)
from neumann.linalg.solve import solve, reduce, _measure
from neumann.linalg.sparse import JaggedArray, csr_matrix


class TestFrame(unittest.TestCase):
    
    def test_frame_main(self):
        # different calls to the creator
        ReferenceFrame(np.eye(3))
        ReferenceFrame(dim=3)
        ReferenceFrame(dim=(3,))
        ReferenceFrame(dim=(10, 3))
        RectangularFrame(dim=3)
        CartesianFrame(dim=3)
        
        # create a random frame and call everything
        angles = np.random.rand(3) * np.pi
        f = ReferenceFrame(dim=3).orient_new('Body', angles, 'XYZ')
        f.dual()
        f.Gram()
        f.is_cartesian()
        f.is_rectangular()
        f.is_independent()
        f.metric_tensor()
        f.axes
        f.axes = f.axes
        f.show()
        f.dcm()
        f.eye(dim=3)
        f.eye(3)
        f.name
        f.orient_new('Body', [0, 0, 90*np.pi/180], 'XYZ')
        f.orient('Body', [0, 0, 90*np.pi/180], 'XYZ')
        f.rotate_new('Body', [0, 0, 90*np.pi/180], 'XYZ')
        f.rotate('Body', [0, 0, 90*np.pi/180], 'XYZ', inplace=False)
        f.rotate('Body', [0, 0, 90*np.pi/180], 'XYZ', inplace=True)
        
        try:
            f.name = 5
        except TypeError:
            pass
        except Exception:
            self.assertTrue(False)
            
        try:
            f.axes = "a"
        except ValueError:
            pass
        except Exception:
            self.assertTrue(False)
            
        try:
            f.axes = np.eye(5)
        except RuntimeError:
            pass
        except Exception:
            self.assertTrue(False)

    def test_frame(self):
        #  GENERAL FRAMES  
        f = ReferenceFrame(dim=3)
        self.assertTrue(f.is_cartesian())
        self.assertTrue(f.is_rectangular())
        f = f.orient_new('Body', [0, 0, 90*np.pi/180], 'XYZ')
        self.assertTrue(f.is_cartesian())
        self.assertTrue(f.is_rectangular())
        f *= 2
        self.assertTrue(f.is_rectangular())
        self.assertFalse(f.is_cartesian())
        # RECTANGULAR FRAMES
        f = RectangularFrame(dim=3)
        self.assertTrue(f.is_rectangular())
        self.assertTrue(f.is_cartesian())
        f *= 2
        self.assertTrue(f.is_rectangular())
        self.assertFalse(f.is_cartesian())
        # CARTESIAN FRAMES
        f = CartesianFrame(dim=3)
        self.assertTrue(f.is_rectangular())
        self.assertTrue(f.is_cartesian())
                        
    def test_dual(self):
        angles = np.random.rand(3) * np.pi
        f = ReferenceFrame(dim=3).orient_new('Body', angles, 'XYZ')
        # the dual of the dual is the original frame
        self.assertTrue(np.allclose(f, f.dual().dual()))
        # the volume of the resiprocal frame is the inverse of the volume
        # of the original frame
        self.assertTrue(np.isclose(1/f.volume(), f.dual().volume()))
        
    def test_frame_ufunc(self):
        """
        Testing against NumPy universal functions.
        """
        # TEST1
        A = CartesianFrame(name='A', axes=np.eye(3))
        v = Vector([1.0, 0.0, 0.0], frame=A)
        A *= 2
        self.assertTrue(type(A) == RectangularFrame)
        self.assertTrue(np.allclose(v, [0.5, 0, 0]))
        A /= 2
        self.assertTrue(type(A) == RectangularFrame)
        self.assertTrue(np.allclose(v, [1.0, 0, 0]))
        A += 1
        self.assertTrue(type(A) == ReferenceFrame)
        self.assertTrue(np.allclose(v.show(CartesianFrame(axes=np.eye(3))), 
                                    [1.0, 0, 0]))
        A -= 1
        self.assertTrue(np.allclose(v, [1.0, 0, 0]))
        # TEST2
        A = CartesianFrame(name='A', axes=np.eye(3))
        v = Vector([1.0, 0.0, 0.0], frame=A)
        A += 1
        A *= 2
        A **= 2
        np.sqrt(A, out=A)
        A /= 2
        A -= 1
        self.assertTrue(np.allclose(v, [1.0, 0, 0]))
        # TEST 3
        A = ReferenceFrame(name='A', axes=np.eye(3))
        v = Vector([1.0, 0.0, 0.0], frame=A)
        A *= 2
        self.assertTrue(np.allclose(v, [0.5, 0, 0]))
        np.sqrt(A, out=A)
        A **= 2
        A /= 2
        self.assertTrue(np.allclose(v, [1.0, 0, 0]))


class TestVector(unittest.TestCase):

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
            
    def test_tr_vector_3(self):
        """
        Transforms a vector to a dual frame and back.
        """
        A = ReferenceFrame(axes=np.eye(3))
        axes = np.eye(3)
        axes[1] = [-1/2, 1/2, 0.]
        B = ReferenceFrame(axes=axes)
        aA = [1.0, 1.0, 1.0]
        vA = Vector(aA, frame=A)
        vB = Vector(vA.show(B), frame=B)
        self.assertTrue(np.allclose(aA, vB.show(A)))


class TestDCM(unittest.TestCase):

    def test_dcm_1(self):
        """
        Vector transformation.
        """
        # old base vectors in old frame
        e1 = np.array([1., 0., 0.])
        e2 = np.array([0., 1., 0.])
        e3 = np.array([0., 0., 1.])

        # new base vectors in old frame
        E1 = np.array([0., 1., 0.])
        E2 = np.array([-1., 0., 0.])
        E3 = np.array([0, 0., 1.])
        
        source = ReferenceFrame(dim=3)
        target = source.orient_new('Body', [0, 0, 90*np.pi/180], 'XYZ')
        DCM = source.dcm(target=target)
        
        # the transpose of DCM transforms the base vectors as column arrays
        assert np.all(np.isclose(DCM.T @ e1, E1, rtol=1e-05, atol=1e-08))
        assert np.all(np.isclose(DCM.T @ e2, E2, rtol=1e-05, atol=1e-08))

        # the DCM transforms the base vectors as row arrays
        assert np.all(np.isclose(e1 @ DCM, E1, rtol=1e-05, atol=1e-08))
        assert np.all(np.isclose(e2 @ DCM, E2, rtol=1e-05, atol=1e-08))

        # transform the complete frame at once
        assert np.all(np.isclose(DCM @ vstack([e1, e2, e3]), vstack([E1, E2, E3]), 
                                 rtol=1e-05, atol=1e-08))
        
    def test_dcm_2(self):
        """
        Equivalent ways of defining the same DCM.
        """
        source = ReferenceFrame(dim=3)
        target = source.orient_new('Body', [0, 0, 90*np.pi/180],  'XYZ')
        DCM_1 = source.dcm(target=target)
        DCM_2 = target.dcm(source=source)
        DCM_3 = target.dcm()  # because source is root
        assert np.all(np.isclose(DCM_1, DCM_2, rtol=1e-05, atol=1e-08))
        assert np.all(np.isclose(DCM_1, DCM_3, rtol=1e-05, atol=1e-08))
        
    def test_dcm_3(self):
        """
        These DCM matrices should admit the identity matrix.
        """
        angles = np.random.rand(3) * np.pi * 2
        source = ReferenceFrame(dim=3)
        target = source.rotate('Body', angles, 'XYZ', inplace=False)
        eye = np.eye(3)
        assert np.all(np.isclose(eye, target.dcm(target=target), rtol=1e-05, atol=1e-08))
        assert np.all(np.isclose(eye, target.dcm(source=target), rtol=1e-05, atol=1e-08))
        assert np.all(np.isclose(eye, source.dcm(target=source), rtol=1e-05, atol=1e-08))
        assert np.all(np.isclose(eye, source.dcm(source=source), rtol=1e-05, atol=1e-08))


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
        
        
class TestSparse(unittest.TestCase):

    def test_csr(self):
        @njit
        def csr_row(csr: csr_matrix, i: int):
            return csr.row(i)
        
        @njit
        def csr_irow(csr: csr_matrix, i: int):
            return csr.irow(i)
        
        @njit
        def csr_data(csr: csr_matrix):
            return csr.data

        np.random.seed = 0
        mat = csr_scipy(np.random.rand(10, 12) > 0.8, dtype=int)
        csr = csr_matrix(mat)
        csr_row(csr, 0)
        csr_irow(csr, 0)
        csr_data(csr)
        csr_matrix.eye(3)
        csr.to_numpy()
        csr.to_scipy()
        csr.row(0)
        csr.irow(0)
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
        jagged = JaggedArray(data, cuts=[3, 3, 3])
        csr_matrix(jagged.to_array()).to_numpy() 
        
    def test_jagged(self):
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        cuts=[3, 3, 4]
        jagged = JaggedArray(data, cuts=cuts)
        jagged.to_csr()
        jagged.to_ak()
        jagged.to_numpy()
        jagged.to_scipy()
        jagged.to_array()
        jagged.to_list()
        np.unique(jagged)
        self.assertTrue(jagged.is_jagged())
        self.assertTrue(np.all(np.isclose(jagged.widths(), cuts)))
        jagged.flatten()
        jagged.shape
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
        cuts=[3, 3, 3]
        jagged = JaggedArray(data, cuts=cuts)
        self.assertFalse(jagged.is_jagged())
        self.assertTrue(np.all(np.isclose(jagged.widths(), cuts)))
        self.assertTrue(np.all(np.isclose(jagged.shape, [3, 3])))
        
        
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


if __name__ == "__main__":

    unittest.main()
