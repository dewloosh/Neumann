import numpy as np
from numpy import ndarray

from .utils import show_vector, show_vectors
from .frame import ReferenceFrame as Frame
from .meta import TensorLike


__all__ = ['Vector']
   

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
    
    >>> from neumann import Vector, ReferenceFrame
    
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
    
    If you want to obtain the components of a vector in a specific
    target frame C, do this:
    
    >>> vB.show(C)
    
    Hence, to create a vector in a target frame C, knowing the components in a 
    source frame A:
    
    >>> vC = Vector(vA.show(C), frame=C)
    
    See Also
    --------
    :class:`~neumann.linalg.array.Array`
    :class:`~neumann.linalg.frame.frame.ReferenceFrame`
    """

    _frame_cls_ = Frame
            
    def dual(self) -> 'Vector':
        """
        Returns the vector described in the dual (or reciprocal) frame.
        """
        a = self.frame.Gram() @ self.array
        return self.__class__(a, frame=self.frame.dual())
    
    def show(self, target: Frame = None, *, dcm:ndarray=None) -> ndarray:
        """
        Returns the components in a target frame. If the target is 
        `None`, the components are returned in the global frame.
        
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
        
    def orient(self, *args, dcm:ndarray=None, **kwargs) -> 'Vector':
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
        