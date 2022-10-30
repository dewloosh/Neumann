# -*- coding: utf-8 -*-
from typing import Callable
from numba import njit, prange
import numpy as np
from numpy import ndarray

__cache = True

__all__ = ['to_range', 'squeeze_if_array', 'config', 'is_none_or_false', 'squeeze']


@njit(nogil=True, parallel=True, cache=__cache)
def _to_range(vals: ndarray, source: ndarray, target: ndarray):
    res = np.zeros_like(vals)
    s0, s1 = source
    t0, t1 = target
    b = (t1- t0) / (s1 - s0)
    a = (t0 + t1) / 2 - b * (s0 + s1) / 2
    for i in prange(res.shape[0]):
        res[i] = a + b * vals[i]
    return res


def to_range(vals: ndarray, *args, source: ndarray, target: ndarray=None, 
             squeeze=False, **kwargs):
    if not isinstance(vals, ndarray):
        vals = np.array([vals,])
    source = np.array([0., 1.]) if source is None else np.array(source)
    target = np.array([-1., 1.]) if target is None else np.array(target)
    if squeeze:
        return np.squeeze(_to_range(vals, source, target))
    else:
        return _to_range(vals, source, target)
    

def squeeze_if_array(arr): 
    return np.squeeze(arr) if isinstance(arr, np.ndarray) else arr


def squeeze(default=True):
    def decorator(fnc: Callable):
        def inner(*args, **kwargs):
            if kwargs.get('squeeze', default):
                res = fnc(*args, **kwargs)
                if isinstance(res, tuple):
                    return list(map(squeeze_if_array, res))
                return squeeze_if_array(res)
            else:
                return fnc(*args, **kwargs)
        inner.__doc__ = fnc.__doc__
        return inner
    return decorator


def config(*args, **kwargs):
    def decorator(fnc: Callable):
        def inner(*args, **kwargs):
            return fnc(*args, **kwargs)
        return inner
    return decorator


def is_none_or_false(a):
    if isinstance(a, bool):
        return not a
    elif a is None:
        return True
    return False