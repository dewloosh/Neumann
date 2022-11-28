# -*- coding: utf-8 -*-
import unittest
from neumann.linalg import ReferenceFrame
import numpy as np
from numpy import vstack


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
        target = source.orient_new('Body', [0, 0, 90*np.pi/180],  'XYZ')
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
        target = source.rotate('Body', angles,  'XYZ', inplace=False)
        eye = np.eye(3)
        assert np.all(np.isclose(eye, target.dcm(target=target), rtol=1e-05, atol=1e-08))
        assert np.all(np.isclose(eye, target.dcm(source=target), rtol=1e-05, atol=1e-08))
        assert np.all(np.isclose(eye, source.dcm(target=source), rtol=1e-05, atol=1e-08))
        assert np.all(np.isclose(eye, source.dcm(source=source), rtol=1e-05, atol=1e-08))


if __name__ == "__main__":

    unittest.main()
