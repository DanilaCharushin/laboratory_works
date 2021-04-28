import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import root

class ConvergenceError(Exception):
    """Class for catching errors if nonlinear equation system cannot be solved due to failure to meet the convergence condition"""
    pass


class NonLinearSystemsSolver(object):
    """Class for solving nonlinear systems by 3 methods: iterations, seidel, newton"""

    # True - print steps in terminal
    DEBUG = False

    @staticmethod
    def solve_by_iterations_method(sys, jac, x0, acc):
        eps = 1 / (10 ** acc)
        n = len(x0)

        s1 = sum(jac(x0)[i][j] for i in range(n) for j in range(n))
        s2 = sum(jac(x0)[j][i] for i in range(n) for j in range(n))
        if s1 >= 1 or s2 >= 1:
            raise ConvergenceError

        x = np.copy(x0)
        x_old = np.copy(x)
        iter_count = 0
        while True:
            A = np.array([
                [jac(x_old)[0][0], jac(x_old)[1][0], 0, 0],
                [0, 0, jac(x_old)[0][0], jac(x_old)[1][0]],
                [jac(x_old)[0][1], jac(x_old)[1][1], 0, 0],
                [0, 0, jac(x_old)[0][1], jac(x_old)[1][1]],
            ], dtype=np.float64)
            b = np.array([-1, 0, 0, -1], dtype=np.float64)
            koef = np.linalg.solve(A, b)

            def prep_sys(x):
                return np.array([
                    x[0] + koef[0] * sys(x)[0] + koef[1] * sys(x)[1],
                    x[1] + koef[2] * sys(x)[0] + koef[3] * sys(x)[1]
                ], dtype=np.float64)

            for i in range(n):
                x[i] = prep_sys(x_old)[i]

            iter_count += 1
            if NonLinearSystemsSolver.DEBUG:
                print(f'Iterations method: step {iter_count}, x = {x}')
            norm = np.array([abs(x[i] - x_old[i]) for i in range(len(x))], dtype=np.float64)
            if np.sqrt(sum(norm[i] ** 2 for i in range(len(norm)))) <= eps:
                break
            else:
                for i in range(n):
                    x_old[i] = x[i]
        for i in range(n):
            x[i] = round(x[i], acc)
        return x, iter_count

    @staticmethod
    def solve_by_seidel_method(sys, jac, x0, acc):
        eps = 1 / (10 ** acc)
        n = len(x0)

        s1 = sum(jac(x0)[i][j] for i in range(n) for j in range(n))
        s2 = sum(jac(x0)[j][i] for i in range(n) for j in range(n))
        if s1 >= 1 or s2 >= 1:
            raise ConvergenceError

        x = np.copy(x0)
        x_old = np.copy(x)
        iter_count = 0
        while True:
            A = np.array([
                [jac(x_old)[0][0], jac(x_old)[1][0], 0, 0],
                [0, 0, jac(x_old)[0][0], jac(x_old)[1][0]],
                [jac(x_old)[0][1], jac(x_old)[1][1], 0, 0],
                [0, 0, jac(x_old)[0][1], jac(x_old)[1][1]],
            ], dtype=np.float64)
            b = np.array([-1, 0, 0, -1], dtype=np.float64)
            koef = np.linalg.solve(A, b)

            def prep_sys(x):
                return np.array([
                    x[0] + koef[0] * sys(x)[0] + koef[1] * sys(x)[1],
                    x[1] + koef[2] * sys(x)[0] + koef[3] * sys(x)[1]
                ], dtype=np.float64)

            for i in range(n):
                x[i] = prep_sys(x)[i]

            iter_count += 1
            if NonLinearSystemsSolver.DEBUG:
                print(f'Iterations method: step {iter_count}, x = {x}')
            norm = np.array([abs(x[i] - x_old[i]) for i in range(len(x))], dtype=np.float64)
            if np.sqrt(sum(norm[i] ** 2 for i in range(len(norm)))) <= eps:
                break
            else:
                for i in range(n):
                    x_old[i] = x[i]
        for i in range(n):
            x[i] = round(x[i], acc)
        return x, iter_count

    @staticmethod
    def solve_by_newton_method(sys, jac, x0, acc):
        eps = 1 / 10**acc
        iter_count = 0
        xnn = np.copy(x0)
        j = jac(x0)
        while True:
            iter_count += 1
            xn = xnn
            xnn = xn - np.linalg.solve(jac(xn), sys(xn))
            if NonLinearSystemsSolver.DEBUG:
                print(f'Newton method: step {iter_count}, x = {xnn}')
            norm = np.array([abs(xnn[i] - xn[i]) for i in range(len(xnn))], dtype=np.float64)
            if np.sqrt(sum(norm[i]**2 for i in range(len(norm)))) <= eps:
                break
        for i in range(len(xnn)):
            xnn[i] = round(xnn[i], acc)
        return xnn, iter_count

    @staticmethod
    def solve_by_integrated_method(sys, jac, x0, acc):
        x = root(sys, x0, jac=jac).x
        for i in range(len(x)):
            x[i] = round(x[i], acc)
        return x


def system(x):
    return np.array([
        np.sin(x[0] + x[1]) - 1.2*x[0] - 0.2,
        x[0]**2 + x[1]**2 - 1
    ], dtype=np.float64)


def jacobian(x):
    return np.array([
        [np.cos(x[0] + x[1]) - 1.2, np.cos(x[0] + x[1])],
        [2*x[0], 2*x[1]]
    ], dtype=np.float64)


def main():
    delta = 0.025
    X, Y = np.meshgrid(np.arange(-2.0, 2.0, delta), np.arange(-1.5, 1.5, delta))
    plt.contour(X, Y, system([X, Y])[0], [0], colors=["#22FF22"])
    plt.contour(X, Y, system([X, Y])[1], [0])
    plt.grid()
    plt.show()
    accuracy = 10
    x0 = np.array([-1, -0.3], dtype=np.float64)
    print("======================================================================")
    print(f'Accuracy = {1 / 10 ** accuracy}')
    print(f'X0 = {x0}')
    print("======================================================================")
    print(f'Integrated: X = {NonLinearSystemsSolver.solve_by_integrated_method(system, jacobian, x0, accuracy)}')

    try:
        x = NonLinearSystemsSolver.solve_by_iterations_method(system, jacobian, x0, accuracy)
        print(f'Iterations method: X = {x[0]}, iterations count = {x[1]}')
    except ConvergenceError:
        print("The convergence condition is not met (iterations method)")
    try:
        x = NonLinearSystemsSolver.solve_by_seidel_method(system, jacobian, x0, accuracy)
        print(f'Seidel method: X = {x[0]}, iterations count = {x[1]}')
    except ConvergenceError:
        print("The convergence condition is not met (seidel method)")
    try:
        x = NonLinearSystemsSolver.solve_by_newton_method(system, jacobian, x0, accuracy)
        print(f'Newton method: X = {x[0]}, iterations count = {x[1]}')
    except ConvergenceError:
        print("The convergence condition is not met (newton method)")

    print("======================================================================")

if __name__ == "__main__":
    main()

