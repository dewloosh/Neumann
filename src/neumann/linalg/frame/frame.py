# -*- coding: utf-8 -*-
import numpy as np
from numpy import ndarray
from sympy.physics.vector import ReferenceFrame as SymPyFrame
from typing import Iterable

from .utils import transpose_dcm_multi
from ..array import Array
from ...array import repeat


class ReferenceFrame(Array):
    """
    A base reference-frame for orthogonal vector spaces. 
    It facilitates tramsformation of tensor-like quantities across 
    different coordinate frames.

    The class is basically an interface on the `ReferenceFrame` class 
    in `sympy.physics.vector`, with a similarly working `orient_new` function.

    Parameters
    ----------
    axes : numpy.ndarray, Optional
        2d numpy array of floats specifying cartesian reference frames.
        Dafault is None.
        
    parent : ReferenceFrame, Optional
        A parent frame in which this the current frame is embedded in.
        Default is False.
    
    dim : int, Optional
        Dimension of the mesh. Deafult is 3.
        
    name : str, Optional
        The name of the frame.
            
    Examples
    --------
    Define a standard Cartesian frame and rotate it around axis 'Z'
    with an amount of 180 degrees:

    >>> A = ReferenceFrame(dim=3)
    >>> B = A.orient_new('Body', [0, 0, np.pi], 'XYZ')

    To create a third frame that rotates from B the way B rotates from A, we
    can do

    >>> A = ReferenceFrame(dim=3)
    >>> C = A.orient_new('Body', [0, 0, 2*np.pi], 'XYZ')

    or we can define it relative to B (this literally makes C to looke 
    in B like B looks in A)

    >>> C = ReferenceFrame(B.axes, parent=B)

    Notes
    -----
    The `polymesh.CartesianFrame` class takes the idea of the reference 
    frame a step further by introducing the idea of the 'origo'. 

    See Also
    --------
    :class:`polymesh.CartesianFrame`
    
    """

    def __init__(self, axes: ndarray = None, parent=None, *args,
                 order: str = 'row', name: str = None, dim: int = None, 
                 **kwargs):
        order = 'C' if order in ['row', 'C'] else 'F'
        try:
            if not isinstance(axes, ndarray):
                if isinstance(dim, Iterable):
                    if len(dim) == 1:
                        axes = np.eye(dim[0])
                    elif len(dim) == 2:
                        axes = repeat(np.eye(dim[-1]), dim[0])
                    else:
                        raise NotImplementedError
                elif isinstance(dim, int):
                    axes = np.eye(dim)
        except Exception as e:
            raise e
        super().__init__(axes, *args, order=order, **kwargs)
        self.name = name
        self.parent = parent
        self._order = 0 if order == 'C' else 1

    @classmethod
    def eye(cls, *args, dim=3, **kwargs) -> 'ReferenceFrame':
        """
        Returns a standard orthonormal frame.
        
        Returns
        -------
        ReferenceFrame
        
        """
        if len(args) > 0 and isinstance(args[0], int):
            dim = args[0]
        return cls(np.eye(dim), *args, **kwargs)

    def root(self) -> 'ReferenceFrame':
        """
        Returns the top level object.
        
        Returns
        -------
        ReferenceFrame
        
        """
        if self.parent is None:
            return self
        else:
            return self.parent.root()

    @property
    def order(self) -> str:
        return 'row' if self._order == 0 else 'col'

    @property
    def axes(self) -> ndarray:
        """
        Returns a matrix, where each row (or column) is the component array
        of a basis vector with respect to the parent frame, or ambient
        space if there is none.
        
        Returns
        -------
        numpy.ndarray
        
        """
        return self._array

    @axes.setter
    def axes(self, value: ndarray):
        """
        Sets the array of the frame.
        """
        if isinstance(value, np.ndarray):
            if value.shape == self._array.shape:
                self._array = value
            else:
                raise RuntimeError("Mismatch in data dimensinons!")
        else:
            raise TypeError("Only numpy arras are supported here!")

    def show(self, target: 'ReferenceFrame' = None) -> ndarray:
        """
        Returns the components of the current frame in a target frame.
        If the target is None, the componants are returned in the ambient frame.
        
        Returns
        -------
        numpy.ndarray
        
        """
        return self.dcm(target=target)

    def dcm(self, *args, target: 'ReferenceFrame' = None,
            source: 'ReferenceFrame' = None, **kwargs) -> ndarray:
        """
        Returns the direction cosine matrix (DCM) of a transformation
        from a source (S) to a target (T) frame. The current frame can be 
        the source or the target, depending on the arguments. 

        If called without arguments, it returns the DCM matrix from the 
        root frame to the current frame (S=root, T=self).

        If `source` is not `None`, then T=self.

        If `target` is not `None`, then S=self.
        
        Parameters
        ----------
        source : 'ReferenceFrame', Optional
            Source frame. Default is None.

        target : 'ReferenceFrame', Optional
            Target frame. Default is None.

        Returns
        -------     
        numpy.ndarray
            DCM matrix from S to T.
            
        Example
        -------
        >>> from neumann.linalg import ReferenceFrame
        >>> import numpy as np
        >>> source = ReferenceFrame(dim=3)
        >>> target = source.orient_new('Body', [0, 0, 90*np.pi/180],  'XYZ')
        >>> DCM = source.dcm(target=target)
        >>> arr_source = np.array([3 ** 0.5 / 2, 0.5, 0])
        >>> arr_target = DCM @ arr_source        

        """
        if source is not None:
            S, T = source.dcm(), self.dcm() 
            return T @ S.T
        elif target is not None:
            S, T = self.dcm(), target.dcm()
            if len(S.shape) == 3:
                return T @ transpose_dcm_multi(S)
            elif len(S.shape) == 2:
                return T @ S.T
            else:
                msg = "There is no transformation rule imlemented for" \
                    " source shape {} and target shape {}"
                raise NotImplementedError(msg.format(S.shape, T.shape))
        # We only get here if the function is called without arguments.
        # The dcm from the ambient frame to the current frame is returned.
        if self.parent is None:
            return self.axes
        else:
            return self.axes @ self.parent.dcm()

    def orient(self, *args, **kwargs) -> 'ReferenceFrame':
        """
        Orients the current frame inplace. 
        See `Referenceframe.orient_new` for the possible arguments.
        
        Parameters
        ----------
        args : tuple, Optional
            A tuple of arguments to pass to the `orientnew` 
            function in `sympy`. 

        kwargs : dict, Optional
            A dictionary of keyword arguments to pass to the 
            `orientnew` function in `sympy`.
            
        Returns
        -------
        ReferenceFrame
        
        See Also
        --------
        :func:`orient_new`

        """
        source = SymPyFrame('source')
        target = source.orientnew('target', *args, **kwargs)
        dcm = np.array(target.dcm(source).evalf()).astype(float)
        self._array = dcm @ self.axes
        return self

    def orient_new(self, *args, name='', **kwargs) -> 'ReferenceFrame':
        """
        Returns a new frame, oriented relative to the called object. 
        The orientation can be provided by all ways supported in 
        `sympy.physics.vector.ReferenceFrame.orientnew`.

        Parameters
        ----------
        name : str
            Name for the new reference frame.

        rot_type : str
            The method used to generate the direction cosine matrix. Supported
            methods are:

            - ``'Axis'``: simple rotations about a single common axis
            - ``'DCM'``: for setting the direction cosine matrix directly
            - ``'Body'``: three successive rotations about new intermediate
              axes, also called "Euler and Tait-Bryan angles"
            - ``'Space'``: three successive rotations about the parent
              frames' unit vectors
            - ``'Quaternion'``: rotations defined by four parameters which
              result in a singularity free direction cosine matrix

        amounts : str
            Expressions defining the rotation angles or direction cosine
            matrix. These must match the ``rot_type``. See examples below for
            details. The input types are:

            - ``'Axis'``: 2-tuple (expr/sym/func, Vector)
            - ``'DCM'``: Matrix, shape(3, 3)
            - ``'Body'``: 3-tuple of expressions, symbols, or functions
            - ``'Space'``: 3-tuple of expressions, symbols, or functions
            - ``'Quaternion'``: 4-tuple of expressions, symbols, or
              functions

        rot_order : str or int, Optional
            If applicable, the order of the successive of rotations. The string
            ``'123'`` and integer ``123`` are equivalent, for example. Required
            for ``'Body'`` and ``'Space'``.
            
        *args : tuple, Optional
            Extra positional arguments forwarded to `sympy.orientnew`.
            Default is None.
            
        **kwargs : dict, Optional
            Extra keyword arguments forwarded to `sympy.orientnew`.
            Default is None.

        Returns
        -------    
        ReferenceFrame
            A new ReferenceFrame object.
            
        See Also
        --------
        :func:`sympy.physics.vector.ReferenceFrame.orientnew`
            
        Examples
        --------
        Define a standard Cartesian frame and rotate it around axis 'Z'
        with 180 degrees:
        
        >>> A = ReferenceFrame(dim=3)
        >>> B = A.orient_new('Body', [0, 0, np.pi], 'XYZ')

        """
        source = SymPyFrame('source')
        target = source.orientnew('target', *args, **kwargs)
        dcm = np.array(target.dcm(source).evalf()).astype(float)
        return self.__class__(axes=dcm, parent=self, name=name)
    
    def rotate(self, *args, inplace:bool=True, **kwargs) -> 'ReferenceFrame':
        """
        Alias for `orient` and `orient_new`, all extra arguments are forwarded.
        
        .. versionadded:: 0.0.4
        
        Parameters
        ----------
        inplace : bool, optional
            If True, transformation is carried out on the instance the function call
            is made upon, otherwise a new frame is returned. Default is True. 
        
        Returns
        -------    
        ReferenceFrame
            An exisitng (if inplace is True) or a new ReferenceFrame object.
        
        See Also
        --------
        :func:`orient`
        :func:`orient_new`
        
        """
        if inplace:
            return self.orient(*args, **kwargs)
        else:
            return self.orient_new(*args, **kwargs)
        
    def rotate_new(self, *args, **kwargs) -> 'ReferenceFrame':
        """
        Alias for `orient_new`, all extra arguments are forwarded.
        
        .. versionadded:: 0.0.4
                
        Returns
        -------    
        ReferenceFrame
            A new ReferenceFrame object.
        
        See Also
        --------
        :func:`orient_new`
        
        """
        return self.orient_new(*args, **kwargs)
