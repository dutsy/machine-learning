# -*- coding: utf-8 -*-
"""
Qussai Firon
Machine Learning EX 1
How to use libraries of matpoltlib & numpy
"""
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

import scipy as sp
import matplotlib.pyplot as plt
import scipy.spatial.distance as sd
import math
import random as r
import numpy as np

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# --------------------- CONSTANTS declarations --------------------------------

arrayx = [2, 5, 10, 25, 50, 100, 500, 1000]
D = 1000
epsilon = math.ldexp(1.0, -53)
kerenlv = sp.arange(0.02, 1.00, 0.02)


# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# --------------- FUNCTIONS / CLASSES declarations ----------------------------

def discrete_gauss(n, g=[0.5, 0.5]):
    """
    discrete_gauss(n, g=[0.5, 0.5])

    Estimates the discrete Gaussian distribution (probability mass function)
    by multiple convolutions with a minimal kernel g.

    :param n: scalar.
           the number of elements of the result (n = 2..1000).
           the functions performs n-2 convolutions to create the result.

     :param g: 1-D array.
           the minimal kernel. Default value is [0.5, 0.5].
           Other kernels of the form [a, 1-a],
           where 0 > a > 1.0 are possible, but they are less effective:
           1. a larger n should be used to be as similar to a Gaussian.
           2. the peak of the result is not centered.

    :return: 1-D array.
          f, the discrete estimate of Gaussian distribution.
         f has n elements.
     """
    if g[0] <= 0 or g[1] >= 1:
        return None
    elif n not in range(2, 1001):
        return None
    elif sum(g) != 1.0:
        return None
    else:
        f = g
    for i in range(n - 2):
        f = sp.convolve(f, g)

    return f


# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# -------------- RUNING the solution to the exercise --------------------------
def show_discrete_gauss(func):
    leng = len(func)
    rang = range(leng)
    title = repr(leng)
    fig = plt.figure()
    fig.suptitle(title, fontsize=20)
    plt.bar(rang, func, bottom=0, color="blue", alpha=0.7)
    plt.show()


# ------------------------------------------------------------------------------
def move_peak_to_center(func):
    func = list(func)
    p = func.index(max(func))
    l = len(func) - 1
    if (p == len(func) // 2):
        return func
    m = min(p, l - p)
    if (m == p):
        return (func[0:2 * p + 1])
    else:
        return (func[l - 2 * (l - p):len(func)])


# ---------------- Question 1c ------------------------------------------------
def que1c():
    for j in arrayx:
        itemf = discrete_gauss(j)
        show_discrete_gauss(itemf)


que1c()


# ----------------- Question 1d -----------------------------------------------
def que1d():
    for j in arrayx:
        itemf = discrete_gauss(j, [0.1, 0.9])
        show_discrete_gauss(itemf)


que1d()


# ---------------- Functions to use for qusetion 1 ----------------------------
def crop(func, size):
    if (size > len(func)):
        return func
    func = list(func)
    p = func.index(max(func))
    if (size % 2 == 0):
        return func[p - size // 2:p + size // 2]
    return func[p - size // 2:p + size // 2 + 1]


def diff_v(v, q):
    if (len(v) != len(q)):
        return None
    s = 0
    for i in range(len(v)):
        s += abs(v[i] - q[i])
    return s


# ---------------- Question 1e ------------------------------------------------
def que1e():
    res1 = []
    res2 = []
    for j in kerenlv:
        n = 999
        func = discrete_gauss(n, [j, 1 - j])
        func = list(func)
        func = move_peak_to_center(func)
        func1 = discrete_gauss(n)
        func1 = crop(func1, len(func))
        func = list(func)
        func1 = (list(func1))
        res1.append(sd.cosine(func, func1))
        res2.append(diff_v(func, func1))

    plt.plot(kerenlv, res1, 'r*-')
    plt.title("Cosine distance  crop method")
    plt.xlabel("kernel")
    plt.ylabel("distance")
    plt.show()
    plt.plot(kerenlv, res2, 'b.-')
    plt.title("|v1-v2| distance  crop method")
    plt.xlabel("kernel")
    plt.ylabel("distance")
    plt.show()


que1e()


# ---------------- Functions to use for qusetion 2 ----------------------------
def dice_roll():
    r.seed()
    return r.randint(1, 6)


def rolls(n=D):
    rollsArr = []
    for i in range(n):
        rollsArr.append(dice_roll())
    return rollsArr

# ----------------- Question 2d -----------------------------------------------
def que2d():
    d = np.arange(1, D + 1)
    s = rolls()
    mean = [np.mean(s[0:i]) for i in d]
    plt.title("Roll vs. the mean of the rolls")
    plt.xlabel("roll")
    plt.ylabel("mean of the rolls")
    plt.plot(d, s, 'b', linewidth=0.5)

    plt.show()

    plt.plot(d, mean, 'b', linewidth=0.5)
    plt.plot(d, [np.mean(range(1, 7)) for i in range(D)], 'r')

    plt.show()


que2d()


# ---------------- Question 2f1 + 2f2------------------------------------------
def que2f():
    x = [rolls(D) for i in range(100)]
    v = []
    e = []
    d = np.arange(1, D + 1, 1)
    v = np.var(x, axis=0)
    e = np.mean(x, axis=0)
    plt.plot(d, v, 'r')
    plt.title("Variance")
    plt.show()
    plt.title("Mean")
    plt.plot(d, e, 'b')
    plt.show()


que2f()
# --------------- END OF FILE -------------------------------------------------