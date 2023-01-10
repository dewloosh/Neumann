from typing import Union, Tuple, Iterable

import numpy as np
from numpy import ndarray
from numba import njit, prange, guvectorize as guv
import sympy as sy
from sympy import symbols, Matrix

from dewloosh.core.tools.alphabet import latinrange

from .meta import TensorLike, ArrayWrapper, FrameLike, ArrayLike
from .exceptions import LinalgOperationInputError, LinalgMissingInputError

__cache = True


def dot(a: Union[TensorLike, ArrayWrapper], b: Union[TensorLike, ArrayWrapper],
        out: Union[TensorLike, ArrayWrapper] = None, frame: FrameLike = None,
        axes: Union[list, tuple] = None) -> Union[TensorLike, ndarray]:
    """
    Returns the doc product of two quantities. The behaviour coincides with NumPy
    when all inputs are arrays and generalizes when they are not, but all inputs 
    must be either all arrays or tensors of some kind. The operation for tensors
    of order 1 and 2 have dedicated implementations, for higher order tensors
    it generalizes to tensor contraction along specified axes.
    
    Parameters
    ----------
    a : TensorLike or ArrayLike
        A tensor or an array.
    b : TensorLike or ArrayLike
        A tensor or an array.
    out : ArrayLike, Optional
        Output argument. This must have the exact kind that would be returned if it was 
        not used. See `numpy.dot` for the details. Only if all inputs are ArrayLike. 
        Default is None.
    frame : FrameLike, Optinal
        The target frame of the output. Only if all inputs are TensorLike. If not specified,
        the returned tensor migh be returned in an arbitrary frame, depending on the inputs.
        Default is None.
    axes : tuple or list, Optional
        The indices along which contraction happens if any of the input tensors have a rank
        higher than 2. Default is None.
        
    Returns
    -------
    TensorLike or ndarray
        An array or a tensor, depending on the inputs.
        
    Examples
    --------
    When working with NumPy arrays, the behaviour coincides with `numpy.dot`. To take the dot
    product of a 2nd order tensor and a vector, use it like this:
    
    >>> from neumann.linalg import ReferenceFrame, Vector, Tensor2
    >>> from neumann.linalg import dot
    >>> frame = ReferenceFrame(np.eye(3))
    >>> A = Tensor2(np.eye(3), frame=frame)
    >>> v = Vector(np.array([1., 0, 0]), frame=frame)
    >>> dot(A, v)
    Array([1., 0., 0.])
    
    For general tensors, you have to specify the axes along which contraction happens:
    
    >>> from neumann.linalg import Tensor
    >>> A = Tensor(np.ones((3, 3, 3, 3)), frame=frame)
    >>> B = Tensor(np.ones((3, 3, 3)), frame=frame)
    >>> dot(A, B, axes=(0, 0)).rank
    5
    """
    if isinstance(a, TensorLike) and isinstance(b, TensorLike):
        ra, rb = a.rank, b.rank
        result = None
        if ra == 1 and rb == 1:
            if out is not None:
                raise LinalgOperationInputError("Parameter 'out' is not allowed with tensors.")
            return np.dot(a.show(), b.show())
        elif ra == 2 and rb == 1:
            arr = (a.array @ b.show(a.frame.dual()).T).T
            result = b.__class__(arr, frame=a.frame)
        elif ra == 1 and rb == 2:
            arr = (a.array.T @ b.show(a.frame.dual()).T).T
            result = a.__class__(arr, frame=a.frame)
        elif ra == 2 and rb == 2:
            g = a.frame.Gram()
            result = a.__class__(a @ g @ b.show(a.frame), frame=a.frame)
        else:
            # If ever implemented, the inner product between a tensor of order n and a
            # tensor of order m is a tensor of order n + m − 2.
            if not axes:
                msg = "The parameter 'axes' is required for tensor contraction of general tensors."
                raise LinalgMissingInputError(msg)
            ia = latinrange(ra, start=ord('a'))
            ib = latinrange(rb, start=ord('a') + ra)
            ax_a, ax_b = axes
            ic = latinrange(1, start=ord('a') + ra + rb)[0]
            ia[ax_a] = ic
            ib[ax_b] = ic
            command = '...' + ''.join(ia) + ',' + '...' + ''.join(ib)
            arr = np.einsum(command, a.show(), b.show(), optimize='greedy')
            result = a.__class__._from_any_input(arr)
        if frame:
            result.frame = frame
        return result
    if frame:
        raise LinalgOperationInputError("Parameter 'frame' is exclusive for tensorial inputs.")
    if not all([isinstance(x, (ndarray, ArrayWrapper, list)) for x in [a, b]]):
        raise TypeError("Invalid types encountered for dot product.")
    inputs = [x._array if isinstance(x, ArrayWrapper) else x for x in [a, b]]
    return np.dot(*inputs, out=out)


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


def is_rectangular_frame(axes: ndarray):
    """
    Returns True if a frame is Cartesian.

    Parameters
    ----------
    axes : numpy.ndarray
        A matrix where the i-th row is the i-th basis vector.
    """
    assert len(axes.shape) == 2, "Input is not a matrix!"
    assert axes.shape[0] == axes.shape[1], "Input is not a square matrix!"
    agram = np.abs(axes @ axes.T)
    return np.isclose(np.trace(agram), np.sum(agram))


def is_normal_frame(axes: ndarray):
    """
    Returns True if a frame is normal, meaning, that it's base vectors
    are all of unit length.

    Parameters
    ----------
    axes : numpy.ndarray
        A matrix where the i-th row is the i-th basis vector.
    """
    return np.allclose(np.linalg.norm(axes, axis=1), 1.0)


def is_orthonormal_frame(axes: ndarray):
    """
    Returns True if a frame is orthonormal.

    Parameters
    ----------
    axes : numpy.ndarray
        A matrix where the i-th row is the i-th basis vector.
    """
    return is_rectangular_frame(axes) and is_normal_frame(axes)


def is_independent_frame(axes: ndarray, tol: float = 0):
    """
    Returns True if a the base vectors of a frame are linearly independent.

    Parameters
    ----------
    axes : numpy.ndarray
        A matrix where the i-th row is the i-th basis vector.
    """
    return np.linalg.det(Gram(axes)) > tol


def is_hermitian(arr:ndarray) -> bool:
    """
    Returns True if the input is a hermitian array.
    """
    shp = arr.shape
    s0 = shp[0]
    return all([s == s0 for s in shp[1:]])


def normalize_frame(axes: ndarray):
    """
    Returns the frame with normalized base vectors.

    Parameters
    ----------
    axes : numpy.ndarray
        A matrix where the i-th row is the i-th basis vector.
    """
    return np.array([normalize(a) for a in axes], dtype=axes.dtype)


def Gram(axes: ndarray):
    """
    Returns the Gram matrix. If a second frame is not provided,
    the Gram matrix of a single frame is returned.

    Parameters
    ----------
    axes : numpy.ndarray
        A matrix where the i-th row is the i-th basis vector.
    """
    return axes @ axes.T


def dual_frame(axes: ndarray) -> ndarray:
    """
    Returns the dual frame of the input.

    Parameters
    ----------
    axes : numpy.ndarray
        A matrix where the i-th row is the i-th basis vector.
    """
    return np.linalg.inv(axes).T


def is_pos_def(arr):
    """
    Returns True if the input is positive definite.
    """
    return np.all(np.linalg.eigvals(arr) > 0)


def is_pos_semidef(arr):
    """
    Returns True if the input is positive semi definite.
    """
    return np.all(np.linalg.eigvals(arr) >= 0)


def random_pos_semidef_matrix(N) -> ndarray:
    """
    Returns a random positive semidefinite matrix of shape (N, N).

    Example
    -------
    >>> from neumann.linalg import random_pos_semidef_matrix, is_pos_semidef
    >>> arr = random_pos_semidef_matrix(2)
    >>> is_pos_semidef(arr)
    True
    """
    A = np.random.rand(N, N)
    return A.T @ A


def random_posdef_matrix(N, alpha: float = 1e-12) -> ndarray:
    """
    Returns a random positive definite matrix of shape (N, N).

    All eigenvalues of this matrix are >= alpha.

    Example
    -------
    >>> from neumann.linalg import random_posdef_matrix, is_pos_def
    >>> arr = random_posdef_matrix(2)
    >>> is_pos_def(arr)
    True
    """
    A = np.random.rand(N, N)
    return A @ A.T + alpha*np.eye(N)


def inv_sym_3x3(m: Matrix, as_adj_det=False):
    P11, P12, P13, P21, P22, P23, P31, P32, P33 = \
        symbols('P_{11} P_{12} P_{13} P_{21} P_{22} P_{23} P_{31} \
                P_{32} P_{33}', real=True)
    Pij = [[P11, P12, P13], [P21, P22, P23], [P31, P32, P33]]
    P = sy.Matrix(Pij)
    detP = P.det()
    adjP = P.adjugate()
    invP = adjP / detP
    subs = {s: r for s, r in zip(sy.flatten(P), sy.flatten(m))}
    if as_adj_det:
        return detP.subs(subs), adjP.subs(subs)
    else:
        return invP.subs(subs)


@njit(nogil=True, parallel=True, cache=__cache)
def vpath(p1: ndarray, p2: ndarray, n: int):
    nD = len(p1)
    dist = p2 - p1
    length = np.linalg.norm(dist)
    s = np.linspace(0, length, n)
    res = np.zeros((n, nD), dtype=p1.dtype)
    d = dist / length
    for i in prange(n):
        res[i] = p1 + s[i] * d
    return res


@njit(nogil=True, cache=__cache)
def linsolve(A, b):
    return np.linalg.solve(A, b)


@njit(nogil=True, cache=__cache)
def inv(A: ndarray):
    return np.linalg.inv(A)


@njit(nogil=True, cache=__cache)
def matmul(A: ndarray, B: ndarray):
    return A @ B


@njit(nogil=True, cache=__cache)
def ATB(A: ndarray, B: ndarray):
    return A.T @ B


@njit(nogil=True, cache=__cache)
def matmulw(A: ndarray, B: ndarray, w: float = 1.0):
    return w * (A @ B)


@njit(nogil=True, cache=__cache)
def ATBA(A: ndarray, B: ndarray):
    return A.T @ B @ A


@njit(nogil=True, cache=__cache)
def ATBAw(A: ndarray, B: ndarray, w: float = 1.0):
    return w * (A.T @ B @ A)


@guv(['(f8[:, :], f8)'], '(n, n) -> ()', nopython=True, cache=__cache)
def det3x3(A, res):
    res = A[0, 0] * A[1, 1] * A[2, 2] - A[0, 0] * A[1, 2] * A[2, 1] \
        - A[0, 1] * A[1, 0] * A[2, 2] + A[0, 1] * A[1, 2] * A[2, 0] \
        + A[0, 2] * A[1, 0] * A[2, 1] - A[0, 2] * A[1, 1] * A[2, 0]


@guv(['(f8[:, :], f8)'], '(n, n) -> ()', nopython=True, cache=__cache)
def det2x2(A, res):
    res = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]


@njit(nogil=True, cache=__cache)
def inv2x2(A):
    res = np.zeros_like(A)
    d = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
    res[0, 0] = A[1, 1] / d
    res[1, 1] = A[0, 0] / d
    res[0, 1] = - A[0, 1] / d
    res[1, 0] = - A[1, 0] / d
    return res


@guv(['(f8[:, :], f8[:, :])'], '(n, n) -> (n, n)', nopython=True, cache=__cache)
def inv2x2u(A, res):
    d = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
    res[0, 0] = A[1, 1] / d
    res[1, 1] = A[0, 0] / d
    res[0, 1] = - A[0, 1] / d
    res[1, 0] = - A[1, 0] / d


@guv(['(f8[:, :], f8[:, :])'], '(n, n) -> (n, n)', nopython=True, cache=__cache)
def adj3x3(A, res):
    res[0, 0] = A[1, 1] * A[2, 2] - A[1, 2] * A[2, 1]
    res[0, 1] = -A[0, 1] * A[2, 2] + A[0, 2] * A[2, 1]
    res[0, 2] = A[0, 1] * A[1, 2] - A[0, 2] * A[1, 1]
    res[1, 0] = -A[1, 0] * A[2, 2] + A[1, 2] * A[2, 0]
    res[1, 1] = A[0, 0] * A[2, 2] - A[0, 2] * A[2, 0]
    res[1, 2] = -A[0, 0] * A[1, 2] + A[0, 2] * A[1, 0]
    res[2, 0] = A[1, 0] * A[2, 1] - A[1, 1] * A[2, 0]
    res[2, 1] = -A[0, 0] * A[2, 1] + A[0, 1] * A[2, 0]
    res[2, 2] = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]


@guv(['(f8[:, :], f8[:, :])'], '(n, n) -> (n, n)', nopython=True, cache=__cache)
def inv3x3u(A, res):
    d = A[0, 0] * A[1, 1] * A[2, 2] - A[0, 0] * A[1, 2] * A[2, 1] \
        - A[0, 1] * A[1, 0] * A[2, 2] + A[0, 1] * A[1, 2] * A[2, 0] \
        + A[0, 2] * A[1, 0] * A[2, 1] - A[0, 2] * A[1, 1] * A[2, 0]
    res[0, 0] = A[1, 1] * A[2, 2] / d - A[1, 2] * A[2, 1] / d
    res[0, 1] = -A[0, 1] * A[2, 2] / d + A[0, 2] * A[2, 1] / d
    res[0, 2] = A[0, 1] * A[1, 2] / d - A[0, 2] * A[1, 1] / d
    res[1, 0] = -A[1, 0] * A[2, 2] / d + A[1, 2] * A[2, 0] / d
    res[1, 1] = A[0, 0] * A[2, 2] / d - A[0, 2] * A[2, 0] / d
    res[1, 2] = -A[0, 0] * A[1, 2] / d + A[0, 2] * A[1, 0] / d
    res[2, 0] = A[1, 0] * A[2, 1] / d - A[1, 1] * A[2, 0] / d
    res[2, 1] = -A[0, 0] * A[2, 1] / d + A[0, 1] * A[2, 0] / d
    res[2, 2] = A[0, 0] * A[1, 1] / d - A[0, 1] * A[1, 0] / d


@njit(nogil=True, cache=__cache)
def inv3x3(A):
    res = np.zeros_like(A)
    det = A[0, 0] * A[1, 1] * A[2, 2] - A[0, 0] * A[1, 2] * A[2, 1] \
        - A[0, 1] * A[1, 0] * A[2, 2] + A[0, 1] * A[1, 2] * A[2, 0] \
        + A[0, 2] * A[1, 0] * A[2, 1] - A[0, 2] * A[1, 1] * A[2, 0]
    res[0, 0] = A[1, 1] * A[2, 2] - A[1, 2] * A[2, 1]
    res[0, 1] = -A[0, 1] * A[2, 2] + A[0, 2] * A[2, 1]
    res[0, 2] = A[0, 1] * A[1, 2] - A[0, 2] * A[1, 1]
    res[1, 0] = -A[1, 0] * A[2, 2] + A[1, 2] * A[2, 0]
    res[1, 1] = A[0, 0] * A[2, 2] - A[0, 2] * A[2, 0]
    res[1, 2] = -A[0, 0] * A[1, 2] + A[0, 2] * A[1, 0]
    res[2, 0] = A[1, 0] * A[2, 1] - A[1, 1] * A[2, 0]
    res[2, 1] = -A[0, 0] * A[2, 1] + A[0, 1] * A[2, 0]
    res[2, 2] = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
    res /= det
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def inv3x3_bulk(A):
    res = np.zeros_like(A)
    for i in prange(A.shape[0]):
        det = A[i, 0, 0] * A[i, 1, 1] * A[i, 2, 2] - A[i, 0, 0] * A[i, 1, 2] \
            * A[i, 2, 1] - A[i, 0, 1] * A[i, 1, 0] * A[i, 2, 2] \
            + A[i, 0, 1] * A[i, 1, 2] * A[i, 2, 0] + A[i, 0, 2] \
            * A[i, 1, 0] * A[i, 2, 1] - A[i, 0, 2] * A[i, 1, 1] * A[i, 2, 0]
        res[i, 0, 0] = A[i, 1, 1] * A[i, 2, 2] - A[i, 1, 2] * A[i, 2, 1]
        res[i, 0, 1] = -A[i, 0, 1] * A[i, 2, 2] + A[i, 0, 2] * A[i, 2, 1]
        res[i, 0, 2] = A[i, 0, 1] * A[i, 1, 2] - A[i, 0, 2] * A[i, 1, 1]
        res[i, 1, 0] = -A[i, 1, 0] * A[i, 2, 2] + A[i, 1, 2] * A[i, 2, 0]
        res[i, 1, 1] = A[i, 0, 0] * A[i, 2, 2] - A[i, 0, 2] * A[i, 2, 0]
        res[i, 1, 2] = -A[i, 0, 0] * A[i, 1, 2] + A[i, 0, 2] * A[i, 1, 0]
        res[i, 2, 0] = A[i, 1, 0] * A[i, 2, 1] - A[i, 1, 1] * A[i, 2, 0]
        res[i, 2, 1] = -A[i, 0, 0] * A[i, 2, 1] + A[i, 0, 1] * A[i, 2, 0]
        res[i, 2, 2] = A[i, 0, 0] * A[i, 1, 1] - A[i, 0, 1] * A[i, 1, 0]
        res[i] /= det
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def inv3x3_bulk2(A):
    res = np.zeros_like(A)
    for i in prange(A.shape[0]):
        res[i] = inv3x3(A[i])
    return res


@njit(nogil=True, cache=__cache)
def normalize(A):
    return A/np.linalg.norm(A)


@njit(nogil=True, parallel=True, cache=__cache)
def normalize2d(A):
    res = np.zeros_like(A)
    for i in prange(A.shape[0]):
        res[i] = normalize(A[i])
    return res


@njit(nogil=True, cache=__cache)
def norm(A):
    return np.linalg.norm(A)


@njit(nogil=True, parallel=True, cache=__cache)
def norm2d(A):
    res = np.zeros(A.shape[0])
    for i in prange(A.shape[0]):
        res[i] = norm(A[i, :])
    return res


@njit(nogil=True, parallel=True, cache=__cache)
def _to_range_1d(vals: ndarray, source: ndarray, target: ndarray):
    res = np.zeros_like(vals)
    s0, s1 = source
    t0, t1 = target
    b = (t1 - t0) / (s1 - s0)
    a = (t0 + t1) / 2 - b * (s0 + s1) / 2
    for i in prange(res.shape[0]):
        res[i] = a + b * vals[i]
    return res


def to_range_1d(vals: ndarray, *args, source: ndarray, target: ndarray = None,
                squeeze=False, **kwargs):
    if not isinstance(vals, ndarray):
        vals = np.array([vals, ])
    source = np.array([0., 1.]) if source is None else np.array(source)
    target = np.array([-1., 1.]) if target is None else np.array(target)
    if squeeze:
        return np.squeeze(_to_range_1d(vals, source, target))
    else:
        return _to_range_1d(vals, source, target)


@njit(nogil=True, parallel=True, cache=__cache)
def _linspace(p0: ndarray, p1: ndarray, N):
    s = p1 - p0
    L = np.linalg.norm(s)
    n = s / L
    djac = L/(N-1)
    step = n * djac
    res = np.zeros((N, p0.shape[0]))
    res[0] = p0
    for i in prange(1, N-1):
        res[i] = p0 + i*step
    res[-1] = p1
    return res


def linspace(start, stop, N):
    if isinstance(start, ndarray):
        return _linspace(start, stop, N)
    else:
        return np.linspace(start, stop, N)


@njit(nogil=True, parallel=True, cache=__cache)
def linspace1d(start, stop, N):
    res = np.zeros(N)
    di = (stop - start) / (N - 1)
    for i in prange(N):
        res[i] = start + i * di
    return res
