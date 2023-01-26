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
    x_k = (max(half_period_len * (i - 1), range_x[0]), half_period_len *
           i, min(half_period_len * (i + 1), range_x[1]))
    print("dupa", x_k)
    if x_k[0] <= x <= x_k[1]:
        print(x, i, (x - x_k[0]) / half_period_len)
        return (x - x_k[0]) / half_period_len
    elif x_k[1] < x <= x_k[2]:
        print(x, i, (x_k[2] - x) / half_period_len, "a")
        return (x_k[2] - x) / half_period_len
    else:
        print(x, i, 0)
        return 0


def e_k_prime(i, x):
    x_k = (half_period_len * (i - 1), half_period_len *
           i, half_period_len * (i + 1))
    if x > x_k[0] and x <= x_k[1]:
        return 1 / half_period_len
    elif x > x_k[1] and x <= x_k[2]:
        return - 1 / half_period_len
    else:
        return 0

# Returns value of B with e_i and e_j arguments, from finite dimensional Vh vector space.
# Vh = span {e_1, e_2, ... e_n}
# Mathematically speaking, B(u, v) is a function, that for given two functions as an
# argument returns real value. Since we are always using e_i functions, we can simply
# use integers as input here, instead of functions.


def B(i, j):
    if i is None:
        def u_x(x): return u_in_zero
        def u_prime_x(x): return 0
        lower_bound = max(range_x[0], (j - 1) * half_period_len)
        upper_bound = min(range_x[1], (j + 1) * half_period_len)
    else:
        def u_x(x): return e_k(i, x)
        def u_prime_x(x): return e_k_prime(i, x)
        lower_bound = max(range_x[0], (i - 1) *
                          half_period_len, (j - 1) * half_period_len)
        upper_bound = min(range_x[1], (i + 1) *
                          half_period_len, (j + 1) * half_period_len)

    def v_x(x): return e_k(j, x)
    def v_prime_x(x): return e_k_prime(j, x)

    return u_x(2) * v_x(2) + integrate_fun(lambda x: (u_prime_x(x) * v_prime_x(x)) - u_x(x) * v_x(x), lower_bound, upper_bound)


def L(i):
    lower_bound = max(range_x[0], (i - 1) * half_period_len)
    upper_bound = min(range_x[1], (i + 1) * half_period_len)
    return integrate_fun(lambda x: e_k(i, x) * sin(x), lower_bound, upper_bound)


def L_shift(i):
    return L(i) - B(None, i)


def main():
    B_matrix = np.fromfunction(np.vectorize(
        lambda x, y: B(x + 1, y + 1)), (n, n))
    L_shift_vector = np.fromfunction(np.vectorize(lambda x: L(x + 1)), (n,))
    coefficient_vector = np.linalg.solve(B_matrix, L_shift_vector)
    x = [i * half_period_len for i in range(n)]
    y = [np.dot(
        coefficient_vector,
        np.fromfunction(
            np.vectorize(lambda x: B(x + 1, i)), (n,)
        )
    ) for i in range(1, n + 1)]
    plt.plot(x, y)
    plt.show()


if __name__ == "__main__":
    main()
