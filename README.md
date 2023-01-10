# **Neumann** - A Python Library for Applied Mathematics in Physical Sciences

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/dewloosh/Neumann/main?labpath=examples%2Flpp.ipynb?urlpath=lab)
[![CircleCI](https://circleci.com/gh/dewloosh/Neumann.svg?style=shield)](https://circleci.com/gh/dewloosh/Neumann)
[![Documentation Status](https://readthedocs.org/projects/neumann/badge/?version=latest)](https://neumann.readthedocs.io/en/latest/?badge=latest)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI](https://badge.fury.io/py/Neumann.svg)](https://pypi.org/project/Neumann)
[![codecov](https://codecov.io/gh/dewloosh/Neumann/branch/main/graph/badge.svg?token=TBI6GG4ECG)](https://codecov.io/gh/dewloosh/Neumann)

> **Warning**
> The library is in a beta stage overall. Wait until version 1.0.

`Neumann` is a Python library that provides tools to formulate and solve problems related to all kinds of scientific disciplines. It is a part of the DewLoosh ecosystem which is designed mainly to solve problems related to computational solid mechanics, but if something is general enough, it ends up here. A good example is the included vector and tensor algebra modules or optimizers, which are applicable in a much broader context than they were designed for.

The most important features:

* Linear Algebra
  * A mechanism that guarantees to maintain the property of objectivity of tensorial quantities.
  * A `ReferenceFrame` class for all kinds of frames, and dedicated `RectangularFrame` and `CartesianFrame` classes as special cases, all NumPy compliant.
  * NumPy compliant classes like `Tensor` and `Vector` to handle various kinds of tensorial quantities efficiently.
  * A `JaggedArray` and a Numba-jittable `csr_matrix` to handle sparse data.

* Operations Research
  * Classes to define and solve linear and nonlinear optimization problems.
    * A `LinearProgrammingProblem` class to define and solve any kind of linear optimization problem.
    * A `BinaryGeneticAlgorithm` class to tackle more complicated optimization problems.

* Graph Theory
  * Algorithms to calculate rooted level structures and pseudo peripheral nodes of a `networkx` graph.

## **Documentation**

The documentation is hosted on [ReadTheDocs](https://Neumann.readthedocs.io/en/latest/).

## **Installation**

`Neumann` can be installed (either in a virtual enviroment or globally) from PyPI using `pip` on Python >= 3.7:

```console
>>> pip install neumann
```

## **Motivating Examples**

### Linear Algebra

Define a reference frame $\mathbf{B}$ relative to the frame $\mathbf{A}$:

```python
>>> from neumann.linalg import ReferenceFrame, Vector, Tensor
>>> A = ReferenceFrame(name='A', axes=np.eye(3))
>>> B = A.orient_new('Body', [0, 0, 90*np.pi/180], 'XYZ', name='B')
```

Get the *DCM matrix* of the transformation between two frames:

```python
>>> B.dcm(target=A)
```

Define a vector $\mathbf{v}$ in frame $\mathbf{A}$ and show the components of it in frame $\mathbf{B}$:

```python
>>> v = Vector([0.0, 1.0, 0.0], frame=A)
>>> v.show(B)
```

Define the same vector in frame $\mathbf{B}$:

```python
>>> v = Vector(v.show(B), frame=B)
>>> v.show(A)
```

### Linear Programming

Solve the following Linear Programming Problem (LPP) with one unique solution:

```python
>>> from neumann.optimize import LinearProgrammingProblem as LPP
>>> from neumann.function import Function, Equality
>>> import sympy as sy
>>> variables = ['x1', 'x2', 'x3', 'x4']
>>> x1, x2, x3, x4 = syms = sy.symbols(variables, positive=True)
>>> obj1 = Function(3*x1 + 9*x3 + x2 + x4, variables=syms)
>>> eq11 = Equality(x1 + 2*x3 + x4 - 4, variables=syms)
>>> eq12 = Equality(x2 + x3 - x4 - 2, variables=syms)
>>> problem = LPP(cost=obj1, constraints=[eq11, eq12], variables=syms)
>>> problem.solve()['x']
array([0., 6., 0., 4.])
```

### NonLinear Programming

Find the minimizer of the Rosenbrock function:

```python
>>> from neumann.optimize import BinaryGeneticAlgorithm
>>> def Rosenbrock(x):
...     a, b = 1, 100
...     return (a-x[0])**2 + b*(x[1]-x[0]**2)**2
>>> ranges = [[-10, 10], [-10, 10]]
>>> BGA = BinaryGeneticAlgorithm(Rosenbrock, ranges, length=12, nPop=200)
>>> BGA.solve()
...
```

## **License**

This package is licensed under the MIT license.
