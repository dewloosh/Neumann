import numpy as np
from numpy import ndarray
from numba import njit, prange, prange

__cache = True


@njit(nogil=True, cache=__cache)
def show_vector(dcm: np.ndarray, arr: np.ndarray):
    """
    Returns the coordinates of a single vector in a frame specified
    by a DCM matrix.
    
    Parameters
    ----------
    dcm : numpy.ndarray
        The dcm matrix of the transformation as a 2d float array.
    arr : numpy.ndarray
        1d float array of coordinates of a single vector.
    
    Returns
    -------      
    numpy.ndarray
        The new coordinates of the vector with the same shape as `arr`.
    """
    return dcm @ arr


@njit(nogil=True, parallel=True, cache=__cache)
def show_vectors(dcm: np.ndarray, arr: np.ndarray):
    """
    Returns the coordinates of multiple vectors in a frame specified
    by a DCM matrix.
    
    Parameters
    ----------
    dcm : numpy.ndarray
        The dcm matrix of the transformation as a 2d float array.
    arr : numpy.ndarray
        2d float array of coordinates of multiple vectors.
    
    Returns
    -------      
    numpy.ndarray
        The new coordinates of the vectors with the same shape as `arr`.
    """
    res = np.zeros_like(arr)
    for i in prange(arr.shape[0]):
        res[i] = dcm @ arr[i, :]
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def show_vectors_multi(dcm: np.ndarray, arr: np.ndarray):
    """
    Returns the coordinates of multiple vectors and multiple DCM matrices.
    
    Parameters
    ----------
    dcm : numpy.ndarray
        The dcm matrix of the transformation as a 3d float array.
    arr : numpy.ndarray
        2d float array of coordinates of multiple vectors.
    
    Returns
    -------      
    numpy.ndarray
        The new coordinates of the vectors with the same shape as `arr`.
    """
    res = np.zeros_like(arr)
    for i in prange(arr.shape[0]):
        res[i] = dcm[i] @ arr[i, :]
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def transpose_dcm_multi(dcm: np.ndarray):
    N = dcm.shape[0]
    res = np.zeros_like(dcm)
    for i in prange(N):
        res[i] = dcm[i].T
    return res


def is_cartesian_frame(axes: ndarray):
    assert len(axes.shape) == 2, "Input is not a matrix!"
    assert axes.shape[0] == axes.shape[1], "Input is not a square matrix!"
    gram = axes @ axes.T
    return np.isclose(np.trace(gram), np.sum(gram))


def is_orthonormal_frame(axes: ndarray):
    if not is_cartesian_frame(axes):
        return False
    else:
        return np.allclose(np.linalg.norm(axes, axis=1), 1.0)