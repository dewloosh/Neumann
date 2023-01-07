import numpy as np
from numpy import ndarray
import numbers

from .utils import show_vector, show_vectors
from .frame import ReferenceFrame as Frame, CartesianFrame
from .meta import TensorLike
from .tensor import SecondOrderTensor
from .exceptions import VectorShapeMismatchError


__all__ = ['Vector']


HANDLED_FUNCTIONS = {}


def implements(numpy_function):
    """
    Register an __array_function__ implementation for Vector 
    objects.
    """
    def decorator(func):
        HANDLED_FUNCTIONS[numpy_function] = func
        return func
    return decorator


class Vector(TensorLike):
    """
    Extends `NumPy`'s ``ndarray`` class to handle arrays with associated
    reference frames. The class also provides a mechanism to transform
    vectors between different frames. Use it like if it was a ``numpy.ndarray`` 
    instance.

    All parameters are identical to those of ``numpy.ndarray``, except that
    this class allows to specify an embedding frame.

    Parameters
    ----------
    args : tuple, Optional
        Positional arguments forwarded to `numpy.ndarray`.
    frame : numpy.ndarray, Optional
        The reference frame the vector is represented by its coordinates.
    kwargs : dict, Optional
        Keyword arguments forwarded to `numpy.ndarray`.

    Examples
    --------
    Import the necessary classes:

    >>> from neumann.linalg import Vector, ReferenceFrame

    Create a default frame in 3d space, and create 2 others, each
    being rotated with 30 degrees around the third axis.

    >>> A = ReferenceFrame(dim=3)
    >>> B = A.orient_new('Body', [0, 0, 30*np.pi/180], 'XYZ')
    >>> C = B.orient_new('Body', [0, 0, 30*np.pi/180], 'XYZ')

    To create a vector in a frame:

    >>> vA = Vector([1.0, 1.0, 0.0], frame=A)

    To create a vector with a relative transformation to another one:

    >>> vB = vA.orient_new('Body', [0, 0, -30*np.pi/180], 'XYZ')

    Use the `array` property to get the componets of a `Vector`:

    >>> vB.array
    Array([1.3660254, 0.3660254, 0.       ])

    If you want to obtain the components of a vector in a specific
    target frame C, do this:

    >>> vB.show(C)
    array([ 1., -1.,  0.])
    
    The reason why the result is represented now as 'array' insted of 'Array'
    as in the previous case is that the Vector class is an array container. When
    you type `vB.array`, what is returned is a wrapped object, an instance of `Array`,
    which is also a class of this library. When you say `vB.show(C)`, a NumPy array
    is returned. Since the `Array` class is a direct subclass of NumPy's `ndarray` class,
    it doesn't really matter and the only difference is the capital first letter. 

    To create a vector in a target frame C, knowing the components in a 
    source frame A:

    >>> vC = Vector(vA.show(C), frame=C)

    See Also
    --------
    :class:`~neumann.linalg.array.Array`
    :class:`~neumann.linalg.frame.frame.ReferenceFrame`
    """

    _frame_cls_ = Frame
    _HANDLED_TYPES_ = (numbers.Number,)

    def dual(self) -> 'Vector':
        """
        Returns the vector described in the dual (or reciprocal) frame.
        """
        a = self.frame.Gram() @ self.array
        return self.__class__(a, frame=self.frame.dual())

    def show(self, target: Frame = None, *, dcm: ndarray = None) -> ndarray:
        """
        Returns the components in a target frame. If the target is 
        `None`, the components are returned in the ambient frame.

        The transformation can also be specified with a proper DCM matrix.

        Parameters
        ----------
        target : numpy.ndarray, Optional
            Target frame.
        dcm : numpy.ndarray, Optional
            The DCM matrix of the transformation.

        Returns
        -------      
        numpy.ndarray
            The components of the vector in a specified frame, or
            the global frame, depending on the arguments.
        """
        if not isinstance(dcm, ndarray):
            if target is None:
                target = self._frame_cls_(dim=self._array.shape[-1])
            dcm = self.frame.dcm(target=target)
        if len(self.array.shape) == 1:
            return show_vector(dcm, self.array)  # dcm @ arr
        else:
            return show_vectors(dcm, self.array)  # dcm @ arr

    def orient(self, *args, dcm: ndarray = None, **kwargs) -> 'Vector':
        """
        Orients the vector inplace. If the transformation is not specified by 'dcm',
        all arguments are forwarded to `orient_new`.

        Parameters
        ----------
        dcm : numpy.ndarray, Optional
            The DCM matrix of the transformation.

        Returns
        -------
        Vector
            The same vector the function is called upon.

        See Also
        --------
        :func:`orient_new`
        """
        if not isinstance(dcm, ndarray):
            fcls = self.__class__._frame_cls_
            dcm = fcls.eye(dim=len(self)).orient_new(*args, **kwargs).dcm()
            self.array = dcm.T @ self._array
        else:
            self.array = np.linalg.inv(dcm) @ self._array
        return self

    def orient_new(self, *args, **kwargs) -> 'Vector':
        """
        Returns a transformed version of the instance.

        Returns
        -------
        Vector
            A new vector.

        See Also
        --------
        :func:`orient`
        """
        fcls = self.__class__._frame_cls_
        dcm = fcls.eye(dim=len(self)).orient_new(*args, **kwargs).dcm()
        array = dcm.T @ self._array
        return Vector(array, frame=self.frame)

    def __array_function__(self, func, types, args, kwargs):
        handled_types = self._HANDLED_TYPES_ + (Vector,)
        if not all(isinstance(x, handled_types) for x in args):
            raise TypeError("If one input is a vector, all other inputs must be vectors!")
            
        if func not in HANDLED_FUNCTIONS:
            arrs = [arg._array for arg in args]
            return func(*arrs, **kwargs)
       
        N = len(args[0])
        for i in range(1, len(args)):
            if not len(args[i]) == N:
                msg = "Input vectors must have the same shape!"
                raise VectorShapeMismatchError(msg)
        return HANDLED_FUNCTIONS[func](*args, **kwargs)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method == 'at':
            raise NotImplementedError("This is currently not implemented")
        out = kwargs.get('out', ())
        for x in inputs + out:
            # Only support operations with instances of _HANDLED_TYPES.
            # Use ArrayLike instead of type(self) for isinstance to
            # allow subclasses that don't override __array_ufunc__ to
            # handle ArrayLike objects.
            if not isinstance(x, self._HANDLED_TYPES_ + (Vector,)):
                raise TypeError("If one input is a vector, all other inputs must be vectors!")

        # Defer to the implementation of the ufunc on unwrapped values.
        frame = CartesianFrame(dim=len(self))
        inputs = tuple(x.show(frame) if isinstance(x, Vector) else x
                       for x in inputs)
        if out:
            kwargs['out'] = tuple(
                x._array if isinstance(x, Vector) else x
                for x in out)
        result = getattr(ufunc, method)(*inputs, **kwargs)

        if type(result) is tuple:
            # multiple return values
            return tuple(type(self)(x, frame=frame) for x in result)
        else:
            # one return value
            return type(self)(result, frame=frame)


@implements(np.dot)
def dot(*args, **kwargs):
    v1, v2 = args[:2]
    return np.dot(v1.show(), v2.show(), **kwargs)


@implements(np.cross)
def cross(*args, **kwargs):
    v1, v2 = args[:2]
    arr = np.cross(v1.show(), v2.show(), **kwargs)
    return Vector(arr, frame=CartesianFrame(dim=len(v1)))


@implements(np.inner)
def inner(*args, **kwargs):
    v1, v2 = args[:2]
    return np.inner(v1.show(), v2.show(), **kwargs)


@implements(np.outer)
def outer(*args, **kwargs):
    v1, v2 = args[:2]
    arr = np.outer(v1.show(), v2.show(), **kwargs)
    return SecondOrderTensor(arr, frame=CartesianFrame(dim=len(v1)))
