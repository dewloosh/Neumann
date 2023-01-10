import numbers

import numpy as np
from numpy import array_repr, array_str

from .meta import TensorLike
from .exceptions import (LinalgInvalidTensorOperationError,
                         TensorShapeMismatchError)
from .utils import dot, cross


__all__ = ['AbstractTensor']


HANDLED_FUNCTIONS = {}


def implements(numpy_function):
    """
    Register an __array_function__ implementation for TensorLike 
    objects.
    """
    def decorator(func):
        HANDLED_FUNCTIONS[numpy_function] = func
        return func
    return decorator


class AbstractTensor(TensorLike):

    _HANDLED_TYPES_ = (numbers.Number,)

    def __imul__(self, other) -> TensorLike:
        if not isinstance(other, numbers.Number):
            raise TypeError("The multiplier must be a scalar.")
        self.array *= other
        return self
    
    def __mul__(self, other) -> TensorLike:
        if not isinstance(other, numbers.Number):
            raise TypeError("The multiplier must be a scalar.")
        arr = self.array * other
        return self.__class__(arr, frame=self.frame)

    def __imatmul__(self, other) -> TensorLike:
        raise LinalgInvalidTensorOperationError("Use a dot product.")
    
    def __matmul__(self, other) -> TensorLike:
        raise LinalgInvalidTensorOperationError("Use a dot product.")

    def __iadd__(self, other) -> TensorLike:
        if isinstance(other, numbers.Number):
            self.array += other
        elif isinstance(other, TensorLike):
            if not self.array.shape == other.array.shape:
                raise TensorShapeMismatchError
            self.array += other.show(self.frame)
        return self

    def __isub__(self, other) -> TensorLike:
        if isinstance(other, numbers.Number):
            self.array += other
        elif isinstance(other, TensorLike):
            if not self.array.shape == other.array.shape:
                raise TensorShapeMismatchError
            self.array -= other.show(self.frame)
        return self

    def __itruediv__(self, other) -> TensorLike:
        if not isinstance(other, numbers.Number):
            raise TypeError("The divider must be a scalar.")
        self.array /= other
        return self
    
    def __truediv__(self, other) -> TensorLike:
        if not isinstance(other, numbers.Number):
            raise TypeError("The multiplier must be a scalar.")
        arr = self.array / other
        return self.__class__(arr, frame=self.frame)

    def __ipow__(self, other) -> TensorLike:
        raise NotImplementedError("This operation is not implemented yet.")
    
    def __pow__(self, other) -> TensorLike:
        raise NotImplementedError("This operation is not implemented yet.")

    def __array_function__(self, func, types, args, kwargs):
        handled_types = self._HANDLED_TYPES_ + (TensorLike,)
        if not all(isinstance(x, handled_types) for x in args):
            raise TypeError("All inputs must be tensors!")
        if func not in HANDLED_FUNCTIONS:
            raise LinalgInvalidTensorOperationError
        else:
            return HANDLED_FUNCTIONS[func](*args, **kwargs)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        msg = """
        NumPy universal functions are not supported by tensors. Try calling 
        this method using the arrays of the tensorial inputs.
        """
        raise LinalgInvalidTensorOperationError(msg)


@implements(np.dot)
def dot_implementation(*args, **kwargs):
    return dot(*args, **kwargs)


@implements(np.cross)
def cross_implementation(*args, **kwargs):
    return cross(*args, **kwargs)


@implements(array_repr)
def array_repr_implementation(*args, **kwargs):
    return array_repr(args[0].array, **kwargs)


@implements(array_str)
def array_str_implementation(*args, **kwargs):
    return array_str(args[0].array, **kwargs)


@implements(np.allclose)
def allclose_implementation(*args, **kwargs):
    return np.allclose(args[0].show(), args[1].show(), **kwargs)


@implements(np.isclose)
def isclose_implementation(*args, **kwargs):
    return np.isclose(args[0].show(), args[1].show(), **kwargs)
