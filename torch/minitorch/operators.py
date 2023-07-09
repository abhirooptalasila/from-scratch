"""
Collection of the core mathematical operators used throughout the code base.
"""

import math
from typing import Callable, Iterable

# ## Task 0.1
#
# Implementation of a prelude of elementary functions.


def mul(x: float, y: float) -> float:
    return x * y


def id(x: float) -> float:
    return x


def add(x: float, y: float) -> float:
    return x + y


def neg(x: float) -> float:
    return -x


def lt(x: float, y: float) -> float:
    return 1.0 if x < y else 0.0


def eq(x: float, y: float) -> float:
    return 1.0 if x == y else 0.0


def max(x: float, y: float) -> float:
    return x if x > y else y


def is_close(x: float, y: float) -> float:
    return 1.0 if math.isclose(x, y, abs_tol=1e-2) else 0.0


def sigmoid(x: float) -> float:
    r"""
    $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$

    (See https://en.wikipedia.org/wiki/Sigmoid_function )

    Calculate as

    $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$

    for stability.
    """
    return 1.0/(1.0 + math.pow(math.e, -x)) if x >= 0 else math.pow(math.e, x)/(1.0 + math.pow(math.e, x))


def relu(x: float) -> float:
    return x if x > 0 else 0.0


EPS = 1e-6


def log(x: float) -> float:
    return math.log(x + EPS)


def exp(x: float) -> float:
    return math.exp(x)


def log_back(x: float, d: float) -> float:
    return d * (1.0 / x)


def inv(x: float) -> float:
    return 1.0 / x


def inv_back(x: float, d: float) -> float:
    return d * (-1.0 / math.pow(x, 2))


def relu_back(x: float, d: float) -> float:
    if x > 0: return d * x
    elif x < 0: return 0.0
    else: raise ValueError("ReLU undefined at 0")


# ## Task 0.3

# Small practice library of elementary higher-order functions.


def map(fn: Callable[[float], float]) -> Callable[[Iterable[float]], Iterable[float]]:
    return lambda ls: [fn(x) for x in ls]


def negList(ls: Iterable[float]) -> Iterable[float]:
    return map(neg)(ls)


def zipWith(
    fn: Callable[[float, float], float]
) -> Callable[[Iterable[float], Iterable[float]], Iterable[float]]:
    return lambda ls1, ls2: [fn(x, y) for x, y in zip(ls1, ls2)]


def addLists(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
    return zipWith(add)(ls1, ls2)


def reduce(
    fn: Callable[[float, float], float], start: float
) -> Callable[[Iterable[float]], float]:
    return lambda ls: start if len(ls) == 0 else reduce(fn, fn(start, ls[0]))(ls[1:])


def sum(ls: Iterable[float]) -> float:
    return reduce(add, 0.0)(ls)

def prod(ls: Iterable[float]) -> float:
    return reduce(mul, 1.0)(ls)
