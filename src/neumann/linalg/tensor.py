from typing import Iterable
from copy import deepcopy as dcopy

import numpy as np
from numpy import ndarray
import sympy as sy

from dewloosh.core.tools.alphabet import latinrange
from .frame import ReferenceFrame as Frame
from .abstract import AbstractTensor
from .top import tr_3333, tr_3333_jit
from .utils import is_hermitian, _transform_tensors2_multi


__all__ = ["Tensor", "Tensor2", "Tensor4", "Tensor2x3", "Tensor4x3"]


class Tensor(AbstractTensor):
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

    Examples
    --------
    Import the necessary classes:

    >>> from neumann.linalg import Tensor, ReferenceFrame
    >>> from numpy.random import rand

    Create a Tensor of order 6 in a frame with random components

    >>> frame = ReferenceFrame(dim=3)
    >>> array = rand(3, 3, 3, 3, 3, 3)
    >>> A = Tensor(array, frame=frame)

    Get the tensor in the dual frame:

    >>> A_dual = A.dual()

    Create an other tensor, in this case a 5th-order one, and calculate their
    generalized dot product, which is a 9th-order tensor:

    >>> from neumann.linalg import dot
    >>> array = rand(3, 3, 3, 3, 3)
    >>> B = Tensor(array, frame=frame)
    >>> C = dot(A, B, axes=[0, 0])
    >>> assert C.rank == (A.rank + B.rank - 2)

    See Also
    --------
    :class:`~neumann.linalg.vector.Vector`
    :class:`~neumann.linalg.frame.ReferenceFrame`
    """

    _frame_cls_ = Frame
    _einsum_params_ = {}

    @classmethod
    def _verify_input(cls, arr: ndarray, *_, **kwargs) -> bool:
        return is_hermitian(arr)

    @classmethod
    def _from_any_input(cls, *args, **kwargs) -> AbstractTensor:
        if cls._verify_input(*args, **kwargs):
            return cls(*args, **kwargs)
        else:
            if not Tensor._verify_input(*args, **kwargs):
                raise ValueError("Invalid input to Tensor class.")
            else:
                return Tensor(*args, **kwargs)

    def dual(self) -> "Tensor2":
        """
        Returns the tensor described in the dual (or reciprocal) frame.
        """
        a = self.transform_components(self.frame.Gram())
        return self.__class__(a, frame=self.frame.dual())

    def transform_components(self, Q: ndarray) -> ndarray:
        """
        Returns the components of the tensor transformed by the matrix Q.
        """
        r = self.rank
        arr = self.array
        args = [Q for _ in range(r)]
        if r not in self.__class__._einsum_params_:
            target = latinrange(r, start=ord("a"))
            source = latinrange(r, start=ord("a") + r)
            terms = [t + s for s, t in zip(source, target)]
            command = ",".join(terms) + "," + "".join(source)
            einsum_path = np.einsum_path(command, *args, arr, optimize="greedy")[0]
            self.__class__._einsum_params_[r] = (command, einsum_path)
        else:
            command, einsum_path = self.__class__._einsum_params_[r]
        return np.einsum(command, *args, arr, optimize=einsum_path)

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
            The components of the tensor in a specified frame, or
            the ambient frame, depending on the arguments.
        """
        if not isinstance(dcm, ndarray):
            if target is None:
                target = Frame(dim=self._array.shape[-1])
            dcm = self.frame.dcm(target=target)
        return self.transform_components(dcm)

    def orient(self, *args, **kwargs) -> "Tensor":
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
        self.array = self.transform_components(dcm.T)
        return self

    def orient_new(self, *args, **kwargs) -> "Tensor":
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
        array = self.transform_components(dcm.T)
        return self.__class__(array, frame=self.frame)

    def copy(self, deep: bool = False, name: str = None) -> "Tensor":
        """
        Returns a shallow or deep copy of this object, depending of the
        argument `deepcopy` (default is False).
        """
        if deep:
            return self.__class__(dcopy(self.array), name=name)
        else:
            return self.__class__(self.array, name=name)

    def deepcopy(self, name: str = None) -> "Tensor":
        """
        Returns a deep copy of the frame.
        """
        return self.copy(deep=True, name=name)


class Tensor2(Tensor):
    """
    A class to handle 2nd-order tensors. Some operations have dedicated implementations
    that provide higher performence utilizing implicit parallelization. Examples include
    the metric tensor, or the stress and strain tensors of elasticity.
    """

    _rank_ = 2

    @classmethod
    def _verify_input(cls, arr: ndarray, *_, bulk: bool = False, **kwargs) -> bool:
        if bulk:
            return len(arr.shape) == 3 and is_hermitian(arr[0])
        else:
            return len(arr.shape) == 2 and is_hermitian(arr)

    def transform_components(self, Q: ndarray) -> ndarray:
        if len(Q.shape) == 3 and len(self.array.shape) == 3:
            return _transform_tensors2_multi(self.array, Q)
        else:
            return Q @ self.array @ Q.T


class Tensor2x3(Tensor2):
    ...


class Tensor4(Tensor):
    """
    A class to handle 4th-order tensors. Some operations have dedicated implementations
    that provide higher performence utilizing implicit parallelization. Examples include
    the piezo-optical tensor, the elasto-optical tensor, the flexoelectric tensor or the
    elasticity tensor.
    """

    _rank_ = 4

    @classmethod
    def _verify_input(cls, arr: ndarray, *_, bulk: bool = False, **kwargs) -> bool:
        if bulk:
            return len(arr.shape) == 5 and is_hermitian(arr[0])
        else:
            return len(arr.shape) == 4 and is_hermitian(arr)


class Tensor4x3(Tensor):
    """
    A class for fourth-order tensors, where each index ranges from 1 to 3.

    Parameters
    ----------
    imap : dict, Optional
        An invertible index map for second-order tensors that assigns to each pair
        of indices a single index. The index map used to switch between 4d and 2d
        representation is inferred from this input. The default is the Voigt indicial map:
            0 : (0, 0)
            1 : (1, 1)
            2 : (2, 2)
            3 : (1, 2)
            4 : (0, 2)
            5 : (0, 1)
    symbolic : bool, Optional
        If True, the tensor is stored in symbolic form, and the components are stored as
        a `SymPy` matrix. Default is False.
    """

    __imap__ = {0: (0, 0), 1: (1, 1), 2: (2, 2), 3: (1, 2), 4: (0, 2), 5: (0, 1)}

    def __init__(self, *args, symbolic: bool = False, imap: dict = None, **kwargs):
        super().__init__(*args, **kwargs)

        if isinstance(imap, dict):
            self.__imap__ = imap

        if symbolic:
            self.transform_components = self._transform_sym_
            self.dtype = object

    @property
    def collapsed(self):
        if self._array is not None:
            return self._array.shape[-1] == 6
        else:
            raise ValueError("There is no data.")

    def expand(self) -> "Tensor4x3":
        """
        Changes the representation of the tensor to 4d.
        """
        if not self.collapsed:
            return self
        T = np.zeros((3, 3, 3, 3), dtype=self._array.dtype)
        m = self._array
        imap = self.imap()
        for ij, ijkl in imap.items():
            T[ijkl] = m[ij]
        self._array = T
        return self

    def collapse(self) -> "Tensor4x3":
        """
        Changes the representation of the tensor to 2d.
        """
        if self.collapsed:
            return self
        m = np.zeros((6, 6), dtype=self._array.dtype)
        T = self._array
        imap = self.imap()
        for ij, ijkl in imap.items():
            m[ij] = T[ijkl]
        self._array = m
        return self

    @classmethod
    def imap(cls, imap1d: dict = None) -> dict:
        """
        Returns a 2d-to-4d index map used to collapse or expand a tensor,
        based on the 1d-to-2d mapping of the class the function is called on,
        or on the first argument, if it is a suitable candidate for an
        index map.
        """
        if imap1d is None:
            imap1d = cls.__imap__
        indices = np.indices((6, 6))
        it = np.nditer([*indices], ["multi_index"])
        imap2d = dict()
        for _ in it:
            i, j = it.multi_index
            imap2d[(i, j)] = imap1d[i] + imap1d[j]
        return imap2d

    @classmethod
    def symbolic(
        cls, *args, base: str = "C_", as_matrix: bool = False, imap: dict = None
    ) -> Iterable:
        """
        Returns a symbolic representation of a 4th order 3x3x3x3 tensor.
        If the argument 'as_matrix' is True, the function returns a 6x6 matrix,
        that unfolds according to the argument 'imap', or if it's not provided,
        the index map of the class the function is called on. If 'imap' is
        provided, it must be a dictionary including exactly 6 keys and
        values. The keys must be integers in the integer range (0, 6), the
        values must be tuples on the integer range (0, 3).
        The default mapping is the Voigt indicial map:
            0 : (0, 0)
            1 : (1, 1)
            2 : (2, 2)
            3 : (1, 2)
            4 : (0, 2)
            5 : (0, 1)
        """
        res = np.zeros((3, 3, 3, 3), dtype=object)
        indices = np.indices((3, 3, 3, 3))
        it = np.nditer([*indices], ["multi_index"])
        for _ in it:
            p, q, r, s = it.multi_index
            if q >= p and s >= r:
                sinds = np.array([p, q, r, s], dtype=np.int16) + 1
                sym = sy.symbols(base + "_".join(sinds.astype(str)))
                res[p, q, r, s] = sym
                res[q, p, r, s] = sym
                res[p, q, s, r] = sym
                res[q, p, s, r] = sym
                res[r, s, p, q] = sym
                res[r, s, q, p] = sym
                res[s, r, p, q] = sym
                res[s, r, q, p] = sym
        if as_matrix:
            mat = np.zeros((6, 6), dtype=object)
            imap = cls.imap(imap) if imap is None else imap
            for ij, ijkl in imap.items():
                mat[ij] = res[ijkl]
            if "sympy" in args:
                res = sy.Matrix(mat)
            else:
                res = mat
        return res

    def transform_components(self, dcm: np.ndarray) -> ndarray:
        """
        Returns the components of the transformed numerical tensor, based on
        the provided direction cosine matrix.
        """
        if self.collapsed:
            self.expand()
            array = tr_3333_jit(self._array, dcm)
            self.collapse()
        else:
            array = tr_3333_jit(self._array, dcm)
        return array

    def _transform_sym_(self, dcm: np.ndarray) -> ndarray:
        """
        Returns the components of the transformed symbolic tensor, based on
        the provided direction cosine matrix.
        """
        if self.collapsed:
            self.expand()
            array = tr_3333(self._array, dcm, dtype=object)
            self.collapse()
        else:
            array = tr_3333(self._array, dcm, dtype=object)
        return array
