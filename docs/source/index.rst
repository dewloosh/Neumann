===========================================================================
**Neumann** - A Python Library for Applied Mathematics in Physical Sciences
===========================================================================

.. admonition:: Important
   :class: caution

   Neumann is in the early stages of it's lifetime, and some concepts may change in 
   the future. If you want long-term stability, wait until version 1.0, which is 
   planned to be released if the core concepts all seem to sit. Nonetheless, the library 
   is well tested with a coverage value above 90%, so if something catches your eye use 
   it with confidence, just don't forget to pin down the version of Neumann in your 
   'requirements.txt'. 

Features
--------

* | Numba-jitted classes and an extendible factory to define and manipulate 
  | vectors and tensors.

* | Classes to define and solve linear and nonlinear optimization
  | problems.

* | A set of array routines for fast prorotyping, including random data creation
  | to assure well posedness, or other properties of test problems.


.. admonition:: Important
   :class: important

   Be aware, that the library uses JIT-compilation through Numba, and as a result,
   first calls to these functions may take longer, but pay off in the long run. 


.. include:: user_guide.md
    :parser: myst_parser.sphinx_

.. toctree::
    :maxdepth: 2
    :glob:
    :caption: Contents
   
    notebooks
    auto_examples/index.rst 

.. toctree::
    :maxdepth: 6
    :glob:
    :hidden:
    :caption: API
   
    api
   
Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
