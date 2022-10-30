# -*- coding: utf-8 -*-
from typing import List, Iterable
import sympy as sy
from sympy import latex
import numpy as np


def gen_Lagrange_1d(*args, x:Iterable=None, i:List[int]=None, xsym: str=None, 
                    fsym: str=None, sym:bool=False, N:int=None, **kwargs) -> dict:
    r"""
    Generates Lagrange polynomials and their derivatives up to 3rd, for appeoximation 
    in 1d space, based on N input pairs of position and value. Geometrical parameters 
    can be numeric or symbolic.

    Parameters
    ----------        
    x : Iterable, Optional
        The locations of the data points. If not specified and `sym=False`, a range 
        of [-1, 1] is assumed and the locations are generated as `np.linspace(-1, 1, N)`, 
        where N is the number of data points. If `sym=True`, the calculation is entirely
        symbolic. Default is None.

    i : List[int]
        If not specified, indices are assumed as [1, ..., N], but this is only relevant 
        for symbolic representation, not the calculation itself, which only cares about 
        the number of data points, regardless of their actual indices.
    
    xsym : str, Optional
        Symbol of the variable in the symbolic representation of the generated functions.
        Default is :math:`x`.

    fsym : str, Optional
        Symbol of the function in the symbolic representation of the generated functions.
        Default is 'f'.

    sym : bool, Optional.
        If True, locations of the data points are left in a symbolic state. This requires
        the inversion of a symbolic matrix, which has some reasonable limitations.
        Default is False.
        
    N : int, Optional
        If neither 'x' nor 'i' is specified, this controls the number of functions to generate.
        Default is None.

    Returns
    -------
    dict
        A dictionary containing the generated functions for the reuested nodes.
        The keys of the dictionary are the indices of the points, the values are
        dictionaries with the following keys and values:

            symbol : the sympy symbol of the function

            0 : the function

            1 : the first derivative as a sympy expression

            2 : the second derivative as a sympy expression

            3 : the third derivative as a sympy expression
            
    Example
    -------
    >>> from neumann.approx.lagrange import gen_Lagrange_1d
    
    To generate approximation functions for a 2-noded line:
    
    >>> gen_Lagrange_1d(x=[-1, 1])
    
    or equivalently
    
    >>> gen_Lagrange_1d(N=2)
    
    To generate the same functions in symbolic form:
    
    >>> gen_Lagrange_1d(i=[1, 2], sym=True)

    Notes
    -----
    Inversion of a heavily symbolic matrix may take quite some time, and is not suggested
    for N > 3. This is one reason why isoparametric finite elements make sense. 
    Fixing the locations as constant real numbers symplifies the process and makes 
    the solution much faster.

    """
    xsym = xsym if xsym is not None else r'x'
    fsym = fsym if fsym is not None else r'\phi'
    module_data = {}
    xvar = sy.symbols(xsym)
    if not isinstance(N, int):
        N = len(x) if x is not None else len(i)
    inds = list(range(1, N + 1)) if i is None else i
    def var_tmpl(i): return r'\Delta_{}'.format(i)
    def var_str(i): return var_tmpl(inds[i])
    coeffs = sy.symbols(', '.join(['c_{}'.format(i+1) for i in range(N)]))
    variables = sy.symbols(', '.join([var_str(i) for i in range(N)]))
    if x is None:
        if xsym is None or not sym:
            x = np.linspace(-1, 1, N)
        else:
            x = sy.symbols(
                ', '.join([xsym + '_{}'.format(i+1) for i in range(N)]))
    poly = sum([c * xvar**i for i, c in enumerate(coeffs)])
    #
    evals = [poly.subs({'x': x[i]}) for i in range(N)]
    A = sy.zeros(N, N)
    for i in range(N):
        A[i, :] = sy.Matrix([evals[i].coeff(c) for c in coeffs]).T
    coeffs_new = A.inv() * sy.Matrix(variables)
    subs = {coeffs[i]: coeffs_new[i] for i in range(N)}
    approx = poly.subs(subs).simplify().expand()
    #
    shp = [approx.coeff(v).factor().simplify() for v in variables]
    #
    def diff(fnc): return fnc.diff(
        xvar).expand().simplify().factor().simplify()
    dshp1 = [diff(fnc) for fnc in shp]
    dshp2 = [diff(fnc) for fnc in dshp1]
    dshp3 = [diff(fnc) for fnc in dshp2]
    #
    for i, ind in enumerate(inds):
        module_data[ind] = {}
        fnc_str = latex(sy.symbols(fsym + '_{}'.format(ind)))
        module_data[ind]['symbol'] = fnc_str
        module_data[ind][0] = shp[i]
        module_data[ind][1] = dshp1[i]
        module_data[ind][2] = dshp2[i]
        module_data[ind][3] = dshp3[i]
    return module_data