import numpy as np
from numpy import ndarray

from dewloosh.core.tools.alphabet import latinrange
from .frame import ReferenceFrame as Frame
from .meta import TensorLike

__all__ = ['Tensor']


class Tensor(TensorLike):
    """
    A class to handle tensors.
            
    Parameters
    ----------
    args : tuple, Optional
        Positional arguments forwarded to `numpy.ndarray`.
    frame : numpy.ndarray, Optional
        The reference frame the vector is represented by its coordinates.
    kwargs : dict, Optional
        Keyword arguments forwarded to `numpy.ndarray`.
    """

    _frame_cls_ = Frame
    
    @classmethod
    def identity(cls, dim=1) -> 'Tensor':
        """
        Returns the identitiy tensor with the specified dimension.
        """
        return cls(np.eye(dim))

    @classmethod
    def eye(cls, dim=1) -> 'Tensor':
        """
        Returns the identitiy tensor with the specified dimension.
        """
        return cls(np.eye(dim))

    def _transform(self, dcm: np.ndarray = None):
        Q = dcm.T
        dim = self.dim
        source = latinrange(dim, start='i')
        target = latinrange(dim, start=ord('i') + dim)
        command = ','.join([t + s for t, s in zip(target, source)]) + \
            ',' + ''.join(source)
        args = [Q for _ in range(dim)]
        return np.einsum(command, *args, self._array, optimize='greedy')
    
    def show(self, target: Frame = None, *args, dcm=None, **kwargs) -> ndarray:
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
        raise NotImplementedError
        if not isinstance(dcm, ndarray):
            if target is None:
                target = Frame(dim=self._array.shape[-1])
            dcm = self.frame.dcm(target=target)
        if len(self.array.shape) == 1:
            return show_vector(dcm, self.array)  # dcm @ arr  
        else:
            return show_vectors(dcm, self.array)  # dcm @ arr
        
    def orient(self, *args, **kwargs) -> 'Tensor':
        """
        Orients the vector inplace. All arguments are forwarded to
        `orient_new`.
        
        Returns
        -------
        Vector
            The same vector the function is called upon.
        
        See Also
        --------
        :func:`orient_new`
        
        """
        fcls = self.__class__._frame_cls_
        dcm = fcls.eye(dim=len(self)).orient_new(*args, **kwargs).dcm()
        self.array = dcm.T @ self._array
        return self

    def orient_new(self, *args, keep_frame=True, **kwargs) -> 'Tensor':
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
        if keep_frame:
            array = dcm.T @ self._array
            return Tensor(array, frame=self.frame)
        else:
            raise NotImplementedError
        

class SecondOrderTensor(Tensor): ...
