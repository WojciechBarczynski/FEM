import sys
import scipy.integrate
import numpy as np
import matplotlib.pyplot as plt
from math import sin


def integrate_fun(function, left_bound, right_bound): return \
    scipy.integrate.quadrature(
        function,
        left_bound,
        right_bound,
        vec_func=False
)[0]


def get_n():
    if (len(sys.argv) > 0):
        match try_to_int(sys.argv[0]):
            case None:
                pass
            case integer:
                return integer
    while True:
        match try_to_int(input('Insert number of elements (n): ')):
            case None:
                continue
            case integer:
                return integer


def try_to_int(string_to_int):
    try:
        return int(string_to_int)
    except ValueError as ex:
        print(f'Passed argument {string_to_int} is not int!')
        return None


n = get_n()
range_x = (0, 2)
half_period_len = (range_x[1] - range_x[0]) / n
period_len = half_period_len * 2
u_in_zero = 2


def e_k(i, x):
    x_k = (half_period_len * (i - 1), half_period_len *
           i, half_period_len * (i + 1))
    if x_k[0] <= x < x_k[1]:
        return (x - x_k[0]) / half_period_len
    elif x_k[1] <= x <= x_k[2]:
        return (x_k[2] - x) / half_period_len
    else:
        return 0


def e_k_prime(i, x):
    x_k = (half_period_len * (i - 1), half_period_len *
           i, half_period_len * (i + 1))
    if x_k[0] <= x < x_k[1]:
        return 1 / half_period_len
    elif x_k[1] <= x <= x_k[2]:
        return - 1 / half_period_len
    else:
        return 0


def B(u, v):
    def u_x(x): return e_k(u, x)
    def u_prime_x(x): return e_k_prime(u, x)
    def v_x(x): return e_k(v, x)
    def v_prime_x(x): return e_k_prime(v, x)

    lower_bound = max(range_x[0], (u - 1) *
                      half_period_len, (v - 1) * half_period_len)
    upper_bound = min(range_x[1], (u + 1) *
                      half_period_len, (v + 1) * half_period_len)

    return u_x(2) * v_x(2) + integrate_fun(lambda x: (u_prime_x(x) * v_prime_x(x)) - u_x(x) * v_x(x), lower_bound, upper_bound)


def L(v):
    lower_bound = max(range_x[0], (v - 1) * half_period_len)
    upper_bound = min(range_x[1], (v + 1) * half_period_len)
    return integrate_fun(lambda x: e_k(v, x) * sin(x), lower_bound, upper_bound)


def L_shift(i):
    return L(i) - 2 * B(0, i)


def get_value_at_x(x, coefficients):
    return sum(coefficients[i] * e_k(i, x) for i in range(n))


def main():
    B_matrix = np.fromfunction(np.vectorize(lambda x, y: B(x+1, y+1)), (n, n))
    L_shift_vector = np.fromfunction(
        np.vectorize(lambda x: L_shift(x+1)), (n,))
    coefficient_vector = np.linalg.solve(B_matrix, L_shift_vector)
    x = [i * half_period_len for i in range(n)]
    y = [get_value_at_x(i, coefficient_vector)
         for i in np.arange(range_x[0], range_x[1], half_period_len)]
    plt.plot(x, y)
    plt.show()


if __name__ == "__main__":
    main()
