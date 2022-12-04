"""
An assortment of 1D Monte Carlo methods. This file consists of a U[a, b] Monte Carlo
distribution, a control variate and antithetic variate Monte Carlo method with U[0,1]
distribution and a naive combination of the two variance reduction methods. Lastly,
a method for an arbitrary distribution by U[0,1] draws using the inverse CDF method.

There is a dictionary of a select PDFs and an dictionary of their inverse CDFs. And
there are a few helper/convenience functions. There is one for variance, covariance
and correlation coefficient. And there is `mcprint` to print out the results of the
MC methods in a more readable format (than as the tuple it returns as).

The variance returned by the MC functions is the variance on the integral not the
integrand, i.e. the variance of the expectation value of the function and not the
function itself where the former is the latter divided by the square root of N, the
number of events.
"""

from typing import Any, Dict, Optional, Union

import numpy as np
from scipy.special import erfinv
from sympy import lambdify, sympify
from vegas import Integrator, RAvgArray, batchintegrand


# Various PDFs and inverse CDFs for `MCd`
inv_cdfs = {
    "uniform": lambda a=0, b=1, N=1: (a + (b - a) * np.random.uniform(size=N)),
    "exponential": lambda lmbda=1, N=1: -np.log(1 - np.random.uniform(size=N)) / lmbda,
    "normal": lambda mu=0, sigma=1, N=1: mu
    + np.sqrt(2) * sigma * erfinv(2 * np.random.uniform(size=N) - 1),
    "logistic": lambda mu=0, s=1, N=1: mu
    + 2 * s * np.arctanh(2 * np.random.uniform(size=N) - 1),
}
pdfs = {
    "uniform": lambda x, a, b: np.where((x < a) | (x > b), 0, 1 / (b - a)),
    "exponential": lambda x, lmbda: lmbda * np.e ** (-lmbda * x),
    "normal": lambda x, mu, sigma: np.e ** (-(((x - mu) / sigma) ** 2) / 2)
    / (np.sqrt(2 * np.pi) * sigma),
    "logistic": lambda x, mu, s: 1 / (4 * s * np.cosh((x - mu) / (2 * s)) ** 2),
}


def mcprint(result: tuple[float, float], text: str = "", sigfigs: int = 5) -> None:
    """
    Pretty print MC results with value +/- standard deviation.

    Parameters:
    result - the tuple result from the MC methods
    text - string to put in front of statement
    sigfigs - number of significant figures for the results
    """
    print(
        text + f"E[f(x)] = {result[0]:.{sigfigs}f} ± {np.sqrt(result[1]):.{sigfigs}f}"
    )


def var(func: str, expval: Optional[float] = None, N: int = int(1e4)) -> float:
    """
    Returns variance for given function for a U[0, 1] distribution.

    Parameters:
    func - function to find variance of as a string
    expval (default None) - expected value of `func`, leave as
        `None` if unknown
    N - number of MC events

    Returns:
    variance
    """
    if expval is None:
        expval = MC(func, N=N, use_vegas=True)[0]

    return MC(str(sympify(func) ** 2), N=N, use_vegas=True)[0] - expval**2


def cov(
    func1: Union[str, np.ndarray],
    func2: Union[str, np.ndarray],
    expval1: Optional[float] = None,
    expval2: Optional[float] = None,
    N: int = int(1e6),
) -> float:
    """
    Returns covariance of two functions or two arrays of random variables of U[0, 1].

    Parameters:
    func1, func2 - Two functions as strings
    expval1, expval2 (defaults None) - expected value of the functions, if unknown, then leave as `None`
    N - number of MC events

    Returns:
    covariance
    """
    expval1 = MC(func1, N=N, use_vegas=True)[0] if expval1 is None else expval1
    expval2 = MC(func2, N=N, use_vegas=True)[0] if expval2 is None else expval2
    prod_expval = MC(str(sympify(func1) * sympify(func2)), N=N, use_vegas=True)[0]

    return prod_expval - expval1 * expval2


def corr(
    func1: str,
    func2: str,
    N: int = int(1e7),
    expval1: Optional[float] = None,
    expval2: Optional[float] = None,
):
    """
    Pearson correlation coefficient between two functions of random variables of U[0, 1].

    Parameters:
    func1, func2 - Two functions as strings
    N - number of MC events
    expval1, expval2 (defaults None) - expected value of the functions, if unknown, then leave as `None`

    Returns:
    correlation coefficient
    """
    return cov(func1, func2, expval1, expval2, N) / np.sqrt(
        var(func1, N=N) * var(func2, N=N)
    )


def MC(
    func: str,
    a: float = 0,
    b: float = 1,
    N: int = int(1e6),
    use_vegas: bool = True,
    vkwargs: Dict[str, Any] = {},
) -> tuple[float, float]:
    """
    1D MC integration of nonnegative function over [a, b] assuming uniform distribution. Option
    to use Vegas to do the MC integration.

    Parameters:
    func - function to integration
    a (default 0) - lower limit of variable
    b (default 1) - upper limit of variable
    N (default int(1e6)) - number of MC events
    use_vegas (default True) - If True, use `vegas` package
    vkwargs (default {}) - Keyword arguments to pass to Vegas

    Returns:
    (integration value, variance)
    """
    # If the function is a  constant, then the expected value is just that constant
    # and there is no variance
    if sympify(func).is_constant():
        return (b - a) * float(sympify(func)), 0

    # Use Vegas for MC:
    if use_vegas:
        func_lmbda = lambdify("x", func)
        integ = Integrator([[a, b]], **vkwargs)
        # Use batchintegration for few times speedup
        # result = integ(batchintegrand(func_lmbda))
        result = integ(func_lmbda)
        # Grab single result
        if isinstance(result, RAvgArray):
            return result[0].mean, result[0].sdev ** 2

        # Return mean and variance
        return result.mean, result.sdev**2

    # Or use my simple version:
    # Otherwise calculate the y's from uniform random x's
    func_lmbda = lambdify("x", func)
    xs = np.random.uniform(a, b, N)
    ys = func_lmbda(xs)
    ys_tot = ys.sum()
    ys_sq_tot = (ys**2).sum()
    expval = (b - a) * ys_tot / N
    expval2 = (b - a) * ys_sq_tot / N

    return expval, (expval2 - expval**2) / N


def MCcv(
    func: str,
    cvfunc: str,
    soln: float,
    coeff: Optional[float] = None,
    N: int = int(1e6),
    cN: int = int(1e6),
    printc: bool = True,
    use_vegas: bool = True,
    vkwargs: Dict[str, Any] = {},
) -> tuple[float, float]:
    """
    1D MC integration of nonnegative function over [0, 1] assuming uniform distribution
    using the control variate method for variance reduction.

    Parameters:
    func - function to integration
    cvfunc - control variate function whose expected value (integration value) is known
    soln - the expected value of `cvfunc`
    N (default int(1e6)) - number of MC events
    cN (default int(1e6)) - number of MC events used when calculating the coefficient (only
        relevant if `coeff=None`.
    printc (default True) - whether to print out value of `c` or not
    use_vegas (default True) - If True, use `vegas` package
    vkwargs (default {}) - Keyword arguments to pass to Vegas

    Returns:
    (integration value, variance)
    """
    # find optimal coefficient as -Cov(func, cvfunc)^2/Var(cvfunc)
    if coeff is None:
        coeff = -cov(func, cvfunc, expval2=soln, N=cN) / var(cvfunc, soln, cN)
        if printc:
            print(f"Using optimized coeffient: c = {coeff:.5f}")

    # Create total function
    totfunc = str(
        (sympify(func) + coeff * (sympify(cvfunc) - soln)).expand().simplify()
    )
    return MC(totfunc, N=N, use_vegas=use_vegas, vkwargs=vkwargs)


def MCav(
    func: str,
    N: int = int(1e6),
    use_vegas: bool = True,
    vkwargs: Dict[str, Any] = {},
) -> tuple[float, float]:
    """
    1D MC integration of nonnegative function over [0,1] assuming uniform distribution
    using the antithetic variates method for variance reduction.

    Parameters:
    func - function to integrate
    N (default int(1e6)) - number of (uncorrelated) MC events
    use_vegas (default True) - If True, use `vegas` package
    vkwargs (default {}) - Keyword arguments to pass to Vegas

    Returns:
    (integration value, variance)
    """
    # Create function representing substitution x -> 1-x
    antfunc = str(sympify(func.replace("x", "(1-x)")))
    # Combined function
    totfunc = str((sympify(func) + sympify(antfunc)) / 2)

    return MC(totfunc, N=N, use_vegas=use_vegas, vkwargs=vkwargs)


def MCd(
    func: str, dist=None, N: int = int(1e6), **dist_kwargs: Optional[float]
) -> tuple[float, float]:
    """
    1D Monte Carlo integration for an arbitrary distribution.

    Parameters:
    func - function to integrate as a string
    dist (default None) - type of distribution. Available distributions are in the dictionary
        `inv_cdfs`. If None is given, then a uniform distribution is assumed. Note that if
        None is given and the limits are given (e.g. a=3, b=10), then `func` must be normalized
        by their difference (e.g. if we want to integrate x**2 from 3 to 10 then our function should
        be (10 - 3)*x**2 because this must cancel out with the uniform distribution PDF: 1/(10 - 3)).
    N (default int(1e6)) - number of MC events
    dist_kwargs - keyword arguments for the parameters of whatever `dist` is. The values of the dictionary
        `inv_cdfs` are lambda function that show the different distributions' parameters if printed out.
    """
    func = lambdify("x", func)

    # grab the correct inverse CDF, the lambda functions already create the uniform distribution
    # values so no need to pass e.g. np.random.uniform(0, 1, N)
    ys = func(inv_cdfs["uniform" if dist is None else dist](**dist_kwargs, N=N))
    ys_avg = np.sum(ys) / N
    ys_sq_avg = np.sum(ys**2) / N

    return ys_avg, (ys_sq_avg - ys_avg**2) / N
