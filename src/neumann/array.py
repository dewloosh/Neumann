# -*- coding: utf-8 -*-
import numpy as np
from numpy import ndarray
from numba import njit, prange
from typing import Union, Tuple
try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable
__cache = True

ArrayOrFloat = Union[float, ndarray, list]


def itype_of_ftype(dtype):
    """
    Returns a matching NumPy integer type to a float type.
    """
    name = np.dtype(dtype).name
    if '32' in name:
        return np.int32
    elif '64' in name:
        return np.int64
    else:
        raise TypeError('Unrecognized float type.')
    

@njit(nogil=True, cache=__cache)
def minmax(a: ndarray) -> Tuple[float]:
    """
    Returns the minimum and maximum values of an array.
    """
    return a.min(), a.max()


def ascont(array: ndarray) -> ndarray:
    """
    Returns the input as contiguous array.
    It is basically a shortcut to `numpy.ascontiguousarray`.
    """
    return np.ascontiguousarray(array)


@njit(nogil=True, cache=__cache)
def clip1d(a: ndarray, a_min: float, a_max: float) -> ndarray:
    """
    Clips the values outside the interval [a_min, a_max] to 
    either a_min or a_max.
    """
    a[a < a_min] = a_min
    a[a > a_max] = a_max
    return a


def atleastnd(a: ndarray, n:int=2, front:bool=True, 
              back:bool=False) -> ndarray:
    """
    Returns an array that is at least 'n' dimensional.
    The required shape is obtained by inserting new axes either
    before or after existing ones. This behaviour can be controlled
    using the parameters 'front' and 'back'. If front is True and back 
    is False, new axes are crated before the first existing data index,
    and the opposite happens in every other case.
    """
    shp = a.shape
    nD = len(shp)
    if nD >= n:
        return a
    else:
        if front and not back:
            newshape = (n - nD) * (1,) + shp
        else:
            newshape = shp + (n - nD) * (1,)
        return np.reshape(a, newshape)


def atleast1d(a: ArrayOrFloat) -> ndarray:
    """Returns an array that is at least 1 dimensional."""
    if not isinstance(a, Iterable):
        a = [a,]
    return np.array(a)


def atleast2d(a: ndarray, **kwargs) -> ndarray:
    """Returns an array that is at least 2 dimensional."""
    return atleastnd(a, 2, **kwargs)


def matrixform(f: ndarray) -> ndarray:
    """Returns an array that is at least 2 dimensional."""
    size = len(f.shape)
    assert size <= 2, "Input array must be at most 2 dimensional."
    if size == 1:
        nV, nC = len(f), 1
        return f.reshape(nV, nC)
    return f


def atleast3d(a: ndarray, **kwargs) -> ndarray:
    """Returns an array that is at least 3 dimensional."""
    return atleastnd(a, 3, **kwargs)


def atleast4d(a: ndarray, **kwargs) -> ndarray:
    """Returns an array that is at least 4 dimensional."""
    return atleastnd(a, 4, **kwargs)


@njit(nogil=True, cache=__cache)
def flatten2dC(a: ndarray) -> ndarray:
    I, J = a.shape
    res = np.zeros(I * J, dtype=a.dtype)
    ind = 0
    for i in range(I):
        for j in range(J):
            res[ind] = a[i, j]
            ind += 1
    return res


@njit(nogil=True, cache=__cache)
def flatten2dF(a: ndarray) -> ndarray:
    I, J = a.shape
    res = np.zeros(I * J, dtype=a.dtype)
    ind = 0
    for j in range(J):
        for i in range(I):
            res[ind] = a[i, j]
            ind += 1
    return res


def flatten2d(a: ndarray, order: str = 'C') -> ndarray:
    """
    Returns a flattened view of `a`.
    """
    if order == 'C':
        return flatten2dC(a)
    elif order == 'F':
        return flatten2dF(a)


def bool_to_float(a: ndarray, true=1.0, false=0.0) -> ndarray:
    """
    Transforms a boolean array to a float array using the specified
    values for `True` and `False`. 
    """
    res = np.full(a.shape, false, dtype=float)
    res[a] = true
    return res


def choice(choices, size, probs=None) -> ndarray:
    """
    Returns a numpy array, whose elements are selected from
    'choices' under probabilities provided with 'probs' (optionally).

    Example
    -------
    >>> N, p = 10, 0.2
    >>> choice([False, True], (N, N), [p, 1-p])
    
    """
    if probs is None:
        probs = np.full((len(choices),), 1/len(choices))
    return np.random.choice(a=choices, size=size, p=probs)


def random_pos_semidef_matrix(N) -> ndarray:
    """
    Returns a random positive semidefinite matrix of shape (N, N).

    Example
    -------
    >>> from neumann.array import random_pos_semidef_matrix
    >>> random_pos_semidef_matrix(3)
    ...
    """
    A = np.random.rand(N, N)
    return A.T @ A


def random_posdef_matrix(N, alpha=1e-12) -> ndarray:
    """
    Returns a random positive definite matrix of shape (N, N).
    
    All eigenvalues of this matrix are >= alpha.

    Example
    -------
    >>> from neumann.array import random_posdef_matrix
    >>> random_posdef_matrix(3, 0.1)
    ...
    """
    A = np.random.rand(N, N)
    return A @ A.T + alpha*np.eye(N)


@njit(nogil=True, parallel=True, cache=__cache)
def repeat(a: ndarray, N:int=1) -> ndarray:
    """
    Repeats an array N-times.
    
    Parameters
    ----------
    a : numpy.ndarray
        Input array.
    
    N : int, Optional
        Number of repetitions. Default is 1.
    
    Returns
    -------
    numpy.ndarray
        A NumPy array with shape (N, a.shape[0], a.shape[1]).
    
    Example
    -------
    
    For example, to generate basis vectors for 10 vectors embedded
    in the same coordinate frame, we need to stack up 10 identical
    identity matrices. This can be done the quickest by:
    
    >>> from neumann.array import repeat
    >>> axes = repeat(np.eye(3), 10)
    
    """
    res = np.zeros((N, a.shape[0], a.shape[1]), dtype=a.dtype)
    for i in prange(N):
        res[i, :, :] = a
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def repeat1d(a: ndarray, N=1) -> ndarray:
    """
    Repeats a 1d array N times.
    """
    M = a.shape[0]
    res = np.zeros(N * M, dtype=a.dtype)
    for i in prange(N):
        res[i*M : (i+1)*M] = a
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def tile(a: ndarray, da: ndarray, N=1) -> ndarray:
    res = np.zeros((N, a.shape[0], a.shape[1]), dtype=a.dtype)
    for i in prange(N):
        res[i, :, :] = a + i*da
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def tile1d(a: ndarray, da: ndarray, N=1) -> ndarray:
    M = a.shape[0]
    res = np.zeros(N * M, dtype=a.dtype)
    for i in prange(N):
        res[i*M : (i+1)*M] = a + i*da
    return res


def indices_of_equal_rows(x: ndarray, y: ndarray, tol=1e-12):
    nX, dX = x.shape
    nY, dY = y.shape
    assert dX == dY, "Input arrays must have identical second dimensions."
    square = nX == nY
    integer = isintegerarray(x) and isintegerarray(y)
    if integer:
        if square:
            inds = np.flatnonzero((x == y).all(1))
            return inds, np.copy(inds)
        else:
            return indices_of_equal_rows_njit(x, y, 0)
    else:
        if square:
            return indices_of_equal_rows_square_njit(x, y, tol)
        else:
            return indices_of_equal_rows_njit(x, y, tol)


@njit(nogil=True, parallel=True, cache=__cache)
def indices_of_equal_rows_njit(x: ndarray, y: ndarray, 
                               tol:float=1e-12):
    R = np.zeros((x.shape[0], y.shape[0]), dtype=x.dtype)
    for i in prange(R.shape[0]):
        for j in prange(R.shape[1]):
            R[i, j] = np.sum(np.abs(x[i] - y[j]))
    return np.where(R <= tol)


@njit(nogil=True, parallel=True, cache=__cache)
def indices_of_equal_rows_square_njit(x: ndarray, y: ndarray,
                                      tol=1e-12):
    n = np.min([x.shape[0], y.shape[0]])
    R = np.zeros((n, n), dtype=x.dtype)
    for i in prange(n):
        for j in prange(n):
            R[i, j] = np.sum(np.abs(x[i] - y[j]))
    return np.where(R <= tol)


@njit(nogil=True, parallel=True, fastmath=True, cache=__cache)
def count_cols(arr: ndarray) -> ndarray:
    """
    Count and return the number of columns for each row in
    the input array.
    """
    n = len(arr)
    res = np.zeros(n, dtype=np.int64)
    for i in prange(n):
        res[i] = len(arr[i])
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def repeat_diagonal_2d(a: ndarray, N:int=2) -> ndarray:
    """
    Assembles a 2d dense block-diagonal matrix by repeating.
    
    Parameters
    ----------
    a : numpy.ndarray
        A 2d array. This is the block that gets repeated.
    
    N : int, Optional
        The number of copies.
        
    Returns
    -------
    numpy.ndarray
        A 2d numpy array.
        
    """
    nR, nC = a.shape[:2]
    res = np.zeros((nR * N, nC * N), dtype=a.dtype)
    for i in prange(N):
        _iR = i * nR
        iR_ = _iR + nR
        _iC = i * nC
        iC_ = _iC + nC
        res[_iR:iR_, _iC:iC_] = a
    return res


# !FIXME : assumes a unique input array
@njit(nogil=True, parallel=True, fastmath=True, cache=__cache)
def find1d(arr, space):
    res = np.zeros(arr.shape, dtype=np.uint64)
    for i in prange(arr.shape[0]):
        res[i] = np.where(arr[i] == space)[0][0]
    return res


# !TODO generalize this up to n dimensions
# !FIXME : assumes a unique input array
@njit(nogil=True, parallel=True, fastmath=True, cache=__cache)
def find2d(arr, space):
    res = np.zeros(arr.shape, dtype=np.uint64)
    for i in prange(arr.shape[0]):
        res[i] = find1d(arr[i], space)
    return res