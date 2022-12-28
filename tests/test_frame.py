import unittest
from neumann.linalg import ReferenceFrame
import numpy as np


class TestFrame(unittest.TestCase):

    def test_frame(self):  
        f = ReferenceFrame(dim=3, cartesian=True)
        self.assertTrue(f.is_cartesian)
        self.assertTrue(f.is_orthonormal)
        
        f = f.orient_new('Body', [0, 0, 90*np.pi/180], 'XYZ')
        self.assertTrue(f.is_cartesian)
        self.assertTrue(f.is_orthonormal)
        
        f *= 2
        self.assertTrue(f.is_cartesian)
        self.assertFalse(f.is_orthonormal)
        
        
        

if __name__ == "__main__":

    unittest.main()