# Adding modules to path, in order to include them properly
import sys
import os
dir = os.path.dirname(__file__)
filepath = os.path.join(dir, '..')
sys.path.insert(0, filepath)
# ----

import warnings
from math import sin
import typing
import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate
from read_utils.read_arguments import read_number_of_elements


# Ignore scipy.integrate accuracy errors
warnings.simplefilter("ignore", scipy.integrate.AccuracyWarning)

BaseFunctionId = typing.NewType('BaseFunctionId', int)


def integrate_fun(function: typing.Callable, left_bound: float, right_bound: float) -> float:
    """Integrates passed function on [left_bound, right_bound] range, using Gaussian quadrature"""
    return \
        scipy.integrate.quadrature(
            function,
            left_bound,
            right_bound,
            vec_func=False,
        )[0]


number_of_elements = read_number_of_elements()
range_x = (0, 2)
half_period_len = (range_x[1] - range_x[0]) / number_of_elements
u_in_zero = 2


def e_k(i: BaseFunctionId, x: float) -> float:
    """Returns value on i-th base function in x"""
    x_k = (half_period_len * (i - 1), half_period_len *
           i, half_period_len * (i + 1))
    if x_k[0] <= x < x_k[1]:
        return (x - x_k[0]) / half_period_len
    elif x_k[1] <= x <= x_k[2]:
        return (x_k[2] - x) / half_period_len
    else:
        return 0


def e_k_prime(i: BaseFunctionId, x: float) -> float:
    """Returns value of i-th base function first derivative 
    (with respect to x) in x"""
    x_k = (half_period_len * (i - 1), half_period_len *
           i, half_period_len * (i + 1))
    if x_k[0] <= x < x_k[1]:
        return 1 / half_period_len
    elif x_k[1] <= x <= x_k[2]:
        return - 1 / half_period_len
    else:
        return 0


def B(u: BaseFunctionId, v: BaseFunctionId) -> float:
    """Returns value of B(e_u, e_v)"""
    def u_x(x): return e_k(u, x)
    def u_prime_x(x): return e_k_prime(u, x)
    def v_x(x): return e_k(v, x)
    def v_prime_x(x): return e_k_prime(v, x)

    lower_bound = max(range_x[0], (u - 1) *
                      half_period_len, (v - 1) * half_period_len)
    upper_bound = min(range_x[1], (u + 1) *
                      half_period_len, (v + 1) * half_period_len)

    return u_x(2) * v_x(2) + integrate_fun(lambda x: (u_prime_x(x) * v_prime_x(x)) - u_x(x) * v_x(x), lower_bound, upper_bound)


def L(v: BaseFunctionId) -> float:
    """Returns value of L(e_v)"""
    lower_bound = max(range_x[0], (v - 1) * half_period_len)
    upper_bound = min(range_x[1], (v + 1) * half_period_len)
    return integrate_fun(lambda x: e_k(v, x) * sin(x), lower_bound, upper_bound)


def L_shift(i: BaseFunctionId) -> float:
    """Returns value of L(v) - B(u_shift, v)"""
    return L(i) - 2 * B(0, i)

def solve_equation() -> np.ndarray:
    B_matrix = np.fromfunction(np.vectorize(lambda x, y: B(  
        x+1, y+1)), (number_of_elements, number_of_elements))
    L_shift_vector = np.fromfunction(
        np.vectorize(lambda x: L_shift(x+1)), (number_of_elements,))
    coefficient_vector = np.linalg.solve(B_matrix, L_shift_vector)
    return coefficient_vector

def get_solution_value_at_x(x: float, coefficients: list) -> float:
    return sum(coefficients[i] * e_k(i, x) for i in range(number_of_elements))


def draw_solution(coefficients: list):
    x = [i * half_period_len for i in range(number_of_elements)]
    y = [get_solution_value_at_x(i, coefficients)
         for i in np.arange(range_x[0], range_x[1], half_period_len)]
    plt.plot(x, y)
    plt.show()


def main():
    coefficient_vector = solve_equation()
    draw_solution(coefficient_vector)


if __name__ == "__main__":
    main()
