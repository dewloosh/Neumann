import numpy as np
from numpy.linalg import LinAlgError
from numba import njit

__cache = True


__all__ = ['solve', 'reduce', 'backsub', 'npsolve']


def solve(A: np.ndarray, B: np.ndarray, presc_bool: np.ndarray = None,
          presc_val: np.ndarray = None, method='numpy', inplace=False):
    if method == 'numpy':
        assert presc_bool is None, \
            "Method '{}' does not support prescribed values.".format(method)
        return npsolve(A, B)
    elif method in ['Jordan', 'Gauss-Jordan']:
        fnc = _Jordan if method == 'Jordan' else _GaussJordan
        try:
            nEQ = len(B)
            if len(B.shape) == 1:
                B = B.reshape((nEQ, 1))
            pre = presc_val is not None
            if not pre:
                presc_bool = np.zeros((nEQ,), dtype=np.int64)
                presc_val = np.zeros((nEQ,), dtype=np.float64)
            if inplace:
                res = backsub(*fnc(A, B, presc_bool, presc_val),
                              presc_bool, presc_val)
            else:
                res = backsub(*fnc(A.copy(), B.copy(), presc_bool, presc_val),
                              presc_bool, presc_val)
            if pre:
                return res
            else:
                return res[0]
        except Exception:
            raise LinAlgError('The matrix is singular!')
    else:
        raise AttributeError("Invalid method name '{}'.".format(method))


def reduce(A: np.ndarray, B: np.ndarray, presc_bool: np.ndarray = None,
           presc_val: np.ndarray = None, method='Gauss-Jordan',
           inplace=False):
    fnc = None
    if method == 'Gauss-Jordan':
        fnc = _GaussJordan
    elif method == 'Jordan':
        fnc = _Jordan
    if fnc is not None:
        try:
            if presc_bool is None:
                nEQ = len(B)
                presc_bool = np.zeros((nEQ,), dtype=np.int64)
                presc_val = np.zeros((nEQ,), dtype=np.float64)
            if inplace:
                return fnc(A, B, presc_bool, presc_val)
            else:
                return fnc(A.copy(), B.copy(), presc_bool, presc_val)
        except Exception:
            raise LinAlgError('The matrix is singular!')
    else:
        raise AttributeError("Invalid method name '{}'.".format(method))


@njit(nogil=True, cache=__cache)
def npsolve(A, b):
    return np.linalg.solve(A, b)


@njit(nogil=True, cache=__cache)
def _Jordan(A: np.ndarray, B: np.ndarray, presc_bool: np.ndarray,
            presc_val: np.ndarray):
    nEQ = len(B)
    for iEQ in range(nEQ):
        if presc_bool[iEQ] == True:
            for iROW in range(iEQ, nEQ):
                B[iROW] -= A[iROW, iEQ] * presc_val[iEQ]
                A[iROW, iEQ] = 0.0
        else:
            pivot = A[iEQ, iEQ]
            if abs(pivot) < 1e-12:
                raise Exception('The matrix is singular!')
            for iROW in range(nEQ):
                factor = A[iROW, iEQ] / pivot
                if iROW == iEQ or abs(factor) < 1e-12:
                    continue
                for iCOL in range(nEQ):
                    A[iROW, iCOL] -= factor * A[iEQ, iCOL]
                B[iROW] -= factor * B[iEQ]
    return A, B


@njit(nogil=True, cache=__cache)
def _GaussJordan(A: np.ndarray, B: np.ndarray, presc_bool: np.ndarray,
                 presc_val: np.ndarray):
    nEQ = len(B)
    for iEQ in range(nEQ):
        if presc_bool[iEQ] == True:
            for iROW in range(iEQ, nEQ):
                B[iROW] -= A[iROW, iEQ] * presc_val[iEQ]
                A[iROW, iEQ] = 0.0
        else:
            if iEQ < nEQ-1:
                pivot = A[iEQ, iEQ]
                if abs(pivot) < 1e-12:
                    raise Exception('The matrix is singular!')
                for iROW in range(iEQ + 1, nEQ):
                    factor = A[iROW, iEQ] / pivot
                    if abs(factor) < 1e-12:
                        continue
                    for iCOL in range(nEQ):
                        A[iROW, iCOL] -= factor * A[iEQ, iCOL]
                    B[iROW] -= factor * B[iEQ]
    return A, B


@njit(nogil=True, cache=__cache)
def backsub(A: np.ndarray, B: np.ndarray, presc_bool: np.ndarray,
            presc_val: np.ndarray):
    nEQ, nRHS = B.shape
    X = np.zeros((nEQ, nRHS), dtype=np.float64)
    R = np.zeros((nEQ, nRHS), dtype=np.float64)
    resid = np.zeros(nRHS, dtype=np.float64)
    for iEQ in range(nEQ - 1, -1, -1):
        pivot = A[iEQ, iEQ]
        resid[:] = B[iEQ]
        if iEQ < nEQ - 1:
            for iCOL in range(iEQ + 1, nEQ):
                resid -= A[iEQ, iCOL] * X[iCOL]
        if presc_bool[iEQ] == True:
            X[iEQ] = presc_val[iEQ]
            R[iEQ] = - resid
        else:
            X[iEQ] = resid / pivot
    return X, R
