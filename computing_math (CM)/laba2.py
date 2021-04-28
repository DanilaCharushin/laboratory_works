import numpy as np
import warnings
from math import isnan, factorial


# контрольные суммы гаусс
# точность

# VARIANT 2
# Kramer, Gauss, Seidel
# 9x1  + 14х2 - 15х3 + 23х4 = 5
# 16x1 + 2х2  - 22х3 + 29х4 = 8
# 18x1 + 20х2 - 3х3  + 32х4 = 9
# 10x1 + 12х2 - 16х3 + 10х4 = 4


class DimensionError(Exception):
    """Class for catching errors if dimensions of matrix are not match"""
    pass


class ConvergenceError(Exception):
    """Class for catching errors if linear equation system cannot be solved
    by Seidel due to failure to meet the convergence condition"""
    pass


class LinearSystemsSolver(object):
    """Class for solving linear systems by 3 methods: Gauss, Kramer, Seidel"""

    # True - print steps in terminal (only for Seidel method)
    DEBUG = False

    @staticmethod
    def solve_by_kramer_method(A, b, accuracy):
        if len(A) != len(b):
            raise DimensionError
        
        n = len(A)
        x = []
        det = np.linalg.det(A)
        count = 2 * factorial(n)
        for i in range(n):
            count += (1 + 2 * factorial(n))
            tmp = np.transpose(np.copy(A))
            tmp[i] = np.copy(b)
            x.append(np.linalg.det(tmp) / det)

        for i in range(n):
            x[i] = round(x[i], accuracy)
        return x, count

    @staticmethod
    def solve_by_gauss_method(A, b, accuracy):
        if len(A) != len(b):
            raise DimensionError
        x = np.transpose(np.append(np.transpose(A), [b], axis=0))
        n = len(x)
        s = np.array([sum(x[i][j] for j in range(n+1)) for i in range(n)], dtype=np.float64)
        current_s = np.copy(s)
        count = 0
        for i in range(n):
            if x[i][i] != 0:
                x[i] = x[i] / x[i][i]
                current_s[i] = current_s[i] / x[i][i]
                count += n
                for j in range(i + 1, n):
                    if x[j][i] != 0:
                        k = (-x[j][i])
                        x[j] = x[j] + k * x[i]
                        current_s[j] = current_s[j] + k * current_s[j]
                        count += (2 * n)
        for i in range(n):
            s[i] = sum(x[i][j] for j in range(n+1))
        print(s)
        print(current_s)
        for i in range(n):
            for j in range(n - i - 2, -1, -1):
                k = (-x[j][n - i - 1]) * x[n - i - 1]
                x[j] = x[j] + k
                count += (2 * n)
        x = np.transpose(x)[n]
        for i in range(len(x)):
            x[i] = round(x[i], accuracy)
        return x, count

    @staticmethod
    def solve_by_seidel_method(A, b, accuracy):
        if len(A) != len(b):
            raise DimensionError

        n = len(A)
        eps = 1 / (10 ** accuracy)
        a_tmp = np.copy(A)
        b_tmp = np.copy(b)

        # normalizing matrix
        for i in range(n):
            max_element = a_tmp[i][i]
            index = i
            for j in range(i, n):
                if abs(max_element) < abs(a_tmp[j][i]):
                    max_element = abs(a_tmp[j][i])
                    index = j
            tmp = np.copy(a_tmp[i])
            a_tmp[i] = a_tmp[index]
            a_tmp[index] = tmp
            tmp = np.copy(b_tmp[i])
            b_tmp[i] = b_tmp[index]
            b_tmp[index] = tmp

        x = np.array([0. for i in range(n)], dtype=np.float64)
        x_old = np.array([0. for i in range(n)], dtype=np.float64)
        d = np.array([0. for i in range(n)], dtype=np.float64)

        iterations_count = 0
        count = 0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            while True:
                if isnan(x[0]):
                    raise ConvergenceError

                for i in range(n):
                    s = b_tmp[i]
                    for j in range(n):
                        if i > j:
                            s -= x[j] * a_tmp[i][j]
                            count += 1
                        elif i != j:
                            s -= x_old[j] * a_tmp[i][j]
                            count += 1
                    x[i] = s / a_tmp[i][i]
                    d[i] = abs(x[i] - x_old[i])
                    count += 2
                iterations_count += 1

                if LinearSystemsSolver.DEBUG:
                    print(f'step {iterations_count}, x = {x}')

                if max(d) < eps:
                    break
                else:
                    for i in range(n):
                        x_old[i] = x[i]

        for i in range(n):
            x[i] = round(x[i], accuracy)
        return x, count, iterations_count

    @staticmethod
    def solve_by_integrated_method(A, b, accuracy):
        if len(A) != len(b):
            raise DimensionError
        x = np.linalg.solve(A, b)
        for i in range(len(A)):
            x[i] = round(x[i], accuracy)
        return x


def main():
    accuracy = 5

    A1 = np.array([
        [9, 14, -15, 23],
        [16, 2, -22, 29],
        [18, 20, -3, 32],
        [10, 12, -16, 10],
    ], dtype=np.float64)
    b1 = np.array([5, 8, 9, 4], dtype=np.float64)

    A2 = np.array([
        [2, 10, 1],
        [10, 1, 1],
        [2, 2, 10]
    ], dtype=np.float64)
    b2 = np.array([12, 13, 14], dtype=np.float64)

    A3 = np.array([
        [13.2, 1.9, 2.3],
        [0.5, -1.4, -9.6],
        [0.8, -7.3, -0.7]
    ], dtype=np.float64)
    b3 = np.array([5.12, 1.5, 5.2], dtype=np.float64)

    A4 = np.array([
        [2.7, 3.3, 1.3],
        [3.5, -1.7, 2.8],
        [4.1, 5.8, -1.7]
    ], dtype=np.float64)
    b4 = np.array([2.1, 1.7, 0.8], dtype=np.float64)

    print(f'Accuracy = {1 / 10**accuracy}')
    try:
        print("=========================1============================")
        print(f'Integrated: x1 = {LinearSystemsSolver.solve_by_integrated_method(A1, b1, accuracy)}')
        x = LinearSystemsSolver.solve_by_gauss_method(A1, b1, accuracy)
        print(f'Gauss: x1 = {x[0]}, arithmetic count = {x[1]}')
        x = LinearSystemsSolver.solve_by_kramer_method(A1, b1, accuracy)
        print(f'Kramer: x1 = {x[0]}, arithmetic count = {x[1]}')
        x = LinearSystemsSolver.solve_by_seidel_method(A1, b1, accuracy)
        print(f'Seidel: x1 = {x[0]}, arithmetic count = {x[1]}, iterations count = {x[2]}')
    except DimensionError:
        print("Dimensions must be the same (A1, b1)")
    except ConvergenceError:
        print("Seidel failed! The convergence condition is not met (A1, b1)")

    try:
        print("=========================2============================")
        print(f'Integrated: x2 = {LinearSystemsSolver.solve_by_integrated_method(A2, b2, accuracy)}')
        x = LinearSystemsSolver.solve_by_gauss_method(A2, b2, accuracy)
        print(f'Gauss: x2 = {x[0]}, arithmetic count = {x[1]}')
        x = LinearSystemsSolver.solve_by_kramer_method(A2, b2, accuracy)
        print(f'Kramer: x2 = {x[0]}, arithmetic count = {x[1]}')
        x = LinearSystemsSolver.solve_by_seidel_method(A2, b2, accuracy)
        print(f'Seidel: x2 = {x[0]}, arithmetic count = {x[1]}, iterations count = {x[2]}')
    except DimensionError:
        print("Dimensions must be the same (A2, b2)")
    except ConvergenceError:
        print("Seidel failed! The convergence condition is not met (A2, b2)")

    try:
        print("=========================3============================")
        print(f'Integrated: x3 = {LinearSystemsSolver.solve_by_integrated_method(A3, b3, accuracy)}')
        x = LinearSystemsSolver.solve_by_gauss_method(A3, b3, accuracy)
        print(f'Gauss: x3 = {x[0]}, arithmetic count = {x[1]}')
        x = LinearSystemsSolver.solve_by_kramer_method(A3, b3, accuracy)
        print(f'Kramer: x3 = {x[0]}, arithmetic count = {x[1]}')
        x = LinearSystemsSolver.solve_by_seidel_method(A3, b3, accuracy)
        print(f'Seidel: x3 = {x[0]}, arithmetic count = {x[1]}, iterations count = {x[2]}')
    except DimensionError:
        print("Dimensions must be the same (A3, b3)")
    except ConvergenceError:
        print("Seidel failed! The convergence condition is not met (A3, b3)")

    try:
        print("=========================4============================")
        print(f'Integrated: x4 = {LinearSystemsSolver.solve_by_integrated_method(A4, b4, accuracy)}')
        x = LinearSystemsSolver.solve_by_gauss_method(A4, b4, accuracy)
        print(f'Gauss: x4 = {x[0]}, arithmetic count = {x[1]}')
        x = LinearSystemsSolver.solve_by_kramer_method(A4, b4, accuracy)
        print(f'Kramer: x4 = {x[0]}, arithmetic count = {x[1]}')
        x = LinearSystemsSolver.solve_by_seidel_method(A4, b4, accuracy)
        print(f'Seidel: x4 = {x[0]}, arithmetic count = {x[1]}, iterations count = {x[2]}')
    except DimensionError:
        print("Dimensions must be the same (A4, b4)")
    except ConvergenceError:
        print("The convergence condition is not met (A4, b4)")


if __name__ == "__main__":
    main()
