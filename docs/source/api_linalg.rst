==============
Linear Algebra
==============

.. _LinAlg:

These classes are meant to provide user friendly, yet high-performance representations of 
Vectors and Tensors in any kind of skew or orthogonal coordinate frame.

The Data Model
==============

The data model is a combination of the two approaches suggested by `NumPy`_ for creating
NumPy-compliant classes. All Vector and Tensor classes are custom array containers that 
wrap a direct subclass of NumPy's ndarray class (see `NumPy custom array containers`_ and
`subclassing ndarray`_ for the details). The double structure allows the container class
to manage and arbitrary array object in the baclground while maintaing a unified interface
to work with directly. For instance the :class:`TopologyArray` class maintains either a NumPy or
an `Awkward`_ array in the background, depending on the input arguments without affecting
the way you interact with an instance. 

.. _NumPy: https://numpy.org/doc/stable/index.html
.. _Awkward: https://awkward-array.org
.. _NumPy custom array containers: https://numpy.org/doc/stable/user/basics.dispatch.html
.. _subclassing ndarray: https://numpy.org/doc/stable/user/basics.dispatch.html

Arrays
======

.. autoclass:: neumann.linalg.meta.ArrayWrapper
    :members:

.. autoclass:: neumann.linalg.sparse.JaggedArray
    :members:

.. autoclass:: neumann.linalg.sparse.csr_matrix
    :members:

Frames
======

.. autoclass:: neumann.linalg.ReferenceFrame
    :members:

.. autoclass:: neumann.linalg.RectangularFrame
    :members:

.. autoclass:: neumann.linalg.CartesianFrame
    :members:

Vectors
=======

.. autoclass:: neumann.linalg.Vector
    :members:

Tensors
=======

.. autoclass:: neumann.linalg.Tensor
    :members:

.. autoclass:: neumann.linalg.Tensor2
    :members:

.. autoclass:: neumann.linalg.Tensor4
    :members:

Routines
========

.. automodule:: neumann.linalg.utils
    :members: 
