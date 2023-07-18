"""
Collection of the core mathematical operators used throughout the code base.
"""

import math
from typing import Callable, Iterable

# ## Task 0.1
#
# Implementation of a prelude of elementary functions.


def mul(x: float, y: float) -> float:
    ":math:`f(x, y) = x * y`"
    return x * y


def id(x: float) -> float:
    ":math:`f(x) = x`"
    return x


def add(x: float, y: float) -> float:
    ":math:`f(x, y) = x + y`"
    return x + y


def neg(x: float) -> float:
    ":math:`f(x) = -x`"
    return -float(x)


def lt(x: float, y: float) -> float:
    ":math:`f(x) =` 1.0 if x is less than y else 0.0"
    return 1.0 if x < y else 0.0


def eq(x: float, y: float) -> float:
    ":math:`f(x) =` 1.0 if x is equal to y else 0.0"
    return 1.0 if x == y else 0.0


def max(x: float, y: float) -> float:
    ":math:`f(x) =` x if x is greater than y else y"
    return x if x > y else y


def is_close(x: float, y: float) -> float:
    ":math:`f(x) = |x - y| < 1e-2` "
    return abs(x - y) < 1e-2


def sigmoid(x: float) -> float:
    r"""
    $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$

    (See https://en.wikipedia.org/wiki/Sigmoid_function )

    Calculate as

    $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$

    for stability.
    """
    return (
        1.0 / (1.0 + math.exp(-x))
        if x >= 0
        else math.exp(x) / (1.0 + math.exp(x))
    )


def relu(x: float) -> float:
    """
    :math:`f(x) =` x if x is greater than 0, else 0

    (See `<https://en.wikipedia.org/wiki/Rectifier_(neural_networks)>`_ .)

    Args:
        x (float): input

    Returns:
        float : relu value
    """
    return x if x > 0 else 0.0


EPS = 1e-6


def log(x: float) -> float:
    ":math:`f(x) = log(x)`"
    return math.log(x + EPS)


def exp(x: float) -> float:
    ":math:`f(x) = e^{x}`"
    return math.exp(x)


def log_back(x: float, d: float) -> float:
    r"If :math:`f = log` as above, compute :math:`d \times f'(x)`"
    return d / (x + EPS)


def inv(x: float) -> float:
    ":math:`f(x) = 1/x`"
    return 1.0 / x


def inv_back(x: float, d: float) -> float:
    r"If :math:`f(x) = 1/x` compute :math:`d \times f'(x)`"
    return - d / (x * x)


def relu_back(x: float, d: float) -> float:
    r"If :math:`f = relu` compute :math:`d \times f'(x)`"
    return d if x > 0 else 0.0


def sigmoid_back(x, d):
    r"If :math:`f = sigmoid` compute :math:`d \times f'(x)`"
    return d * sigmoid(x) * (1 - sigmoid(x))


def exp_back(x, d):
    r"If :math:`f = exp` compute :math:`d \times f'(x)`"
    return d * exp(x)


def map(fn: Callable[[float], float]) -> Callable[[Iterable[float]], Iterable[float]]:
    """
    Higher-order map.

    .. image:: figs/Ops/maplist.png


    See `<https://en.wikipedia.org/wiki/Map_(higher-order_function)>`_

    Args:
        fn (one-arg function): Function from one value to one value.

    Returns:
        function : A function that takes a list, applies `fn` to each element, and returns a
        new list
    """
    return lambda ls: [fn(x) for x in ls]


def negList(ls: Iterable[float]) -> Iterable[float]:
    "Use :func:`map` and :func:`neg` to negate each element in `ls`"
    return map(neg)(ls)


def zipWith(
    fn: Callable[[float, float], float]
) -> Callable[[Iterable[float], Iterable[float]], Iterable[float]]:
    """
    Higher-order zipwith (or map2).

    .. image:: figs/Ops/ziplist.png

    See `<https://en.wikipedia.org/wiki/Map_(higher-order_function)>`_

    Args:
        fn (two-arg function): combine two values

    Returns:
        function : takes two equally sized lists `ls1` and `ls2`, produce a new list by
        applying fn(x, y) on each pair of elements.

    """
    return lambda ls1, ls2: [fn(x, y) for x, y in zip(ls1, ls2)]


def addLists(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
    "Add the elements of `ls1` and `ls2` using :func:`zipWith` and :func:`add`"
    return zipWith(add)(ls1, ls2)


def reduce(
    fn: Callable[[float, float], float], start: float
) -> Callable[[Iterable[float]], float]:
    r"""
    Higher-order reduce.

    .. image:: figs/Ops/reducelist.png


    Args:
        fn (two-arg function): combine two values
        start (float): start value :math:`x_0`

    Returns:
        function : function that takes a list `ls` of elements
        :math:`x_1 \ldots x_n` and computes the reduction :math:`fn(x_3, fn(x_2,
        fn(x_1, x_0)))`
    """
    def _reduce(ls):
        if len(ls) == 0:
            return start
        elif len(ls) == 1:
            return fn(start, ls[0])
        else:
            return fn(_reduce(ls[:-1]), ls[-1])
    return _reduce


def sum(ls: Iterable[float]) -> float:
    "Sum up a list using :func:`reduce` and :func:`add`."
    return reduce(add, 0.0)(ls)


def prod(ls: Iterable[float]) -> float:
    "Product of a list using :func:`reduce` and :func:`mul`."
    return reduce(mul, 1.0)(ls)
