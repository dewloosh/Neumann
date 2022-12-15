import unittest
import numpy as np
from numba import njit
from scipy.sparse import csr_matrix as csr_scipy

from neumann.linalg.sparse import JaggedArray, csr_matrix


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
        
        
if __name__ == "__main__":

    unittest.main()
