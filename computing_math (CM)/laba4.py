import numpy as np
import matplotlib.pyplot as plt
from math import factorial
from sympy import *
from scipy.interpolate import lagrange as lg
from scipy.interpolate import interp1d as interp
from numpy import linalg as LA


def func(x):
    # return 2*2.718281828**x + 3*x + 1
    return x**3 - x**2 + 2*x + 3


class ApproximateErrorTester(object):
    @staticmethod
    def test_by_taylor_series(func, xx, a=0, b=1, points_number=5, max_order=1, x0_number=3):
        x = np.linspace(a, b, points_number)
        c = np.linspace(a, b, x0_number)

        def deriv(x0, n):
            return float(diff(func, xx, n).evalf(subs={xx: x0}))

        def T(x0, x, n):
            if n == 0:
                return deriv(x0, n)
            return T(x0, x, n-1) + deriv(x0, n) * (x - x0)**n / factorial(n)

        figs = []
        axes = []
        figs_err = []
        axes_err = []
        for i in range(x0_number):
            fig, ax = plt.subplots()
            fig_err, ax_err = plt.subplots()
            figs.append(fig)
            figs_err.append(fig_err)
            axes.append(ax)
            axes_err.append(ax_err)

        X = np.linspace(a, b, 1000)
        Y1 = np.array([deriv(X[n], 0) for n in range(1000)], dtype=np.float64)
        Y2 = np.array([deriv(x[n], 0) for n in range(points_number)], dtype=np.float64)
        for i in range(x0_number):
            axes[i].axvline(c[i], c='k', label="c")
            axes_err[i].axvline(c[i], c='k', label="c")
            axes[i].plot(X, Y1, '-r', label="func", lw="4")
            axes[i].plot(x, Y2, 'ob', ms="8", label="points")
            axes[i].set_title(f'Taylor | Graphics | c={round(c[i], 2)}')
            axes_err[i].set_title(f'Taylor | Error | c={round(c[i], 2)}')
            axes_err[i].set_xlabel("x")
            axes_err[i].set_ylabel("y")
            axes[i].set_xlabel("x")
            axes[i].set_ylabel("y")

        color = ['b', 'g', 'c', 'y']
        for i in range(1, max_order + 1):
            for j in range(x0_number):
                err = np.array([round(1 - T(c[j], x[n], i) / Y2[n], 3) for n in range(points_number)], dtype=np.float64)
                Y = np.array([T(c[j], x[n], i) for n in range(points_number)], dtype=np.float64)
                axes[j].plot(x, Y, f'--{color[i%len(color)]}', lw="3", label=f'n={i}')
                axes_err[j].plot(x, err, f'-{color[i%len(color)]}', label=f'n={i}')
                axes[j].legend()
                axes[j].grid()
                axes_err[j].legend()
                axes_err[j].grid()

        plt.show()
        err = []
        f = np.array([deriv(c[j], 0) for j in range(x0_number)], dtype=np.float64)
        for i in range(x0_number):
            data = []
            for n in range(1, max_order + 1):
                q = np.array([T(c[i], c[j], n) for j in range(x0_number)], dtype=np.float64)
                data.append(round(LA.norm((f-q)/f), 3))
            err.append(data)

        return c, err

    @staticmethod
    def test_by_lagrange(func, a=0, b=1, points_number=5):
        x = np.linspace(a, b, points_number)
        X = np.linspace(a, b, 1000)
        y = func(x)
        p = lg(x, y)
        fig, ax = plt.subplots()
        fig_err, ax_err = plt.subplots()

        ax.plot(X, func(X), '-r', label="func", lw="4")
        ax.plot(x, y, 'ob', ms="8", label="points")
        ax.plot(X, p(X), f'--g', lw="3", label=f'n={points_number-1}')
        ax.set_title("Lagrange | Graphics")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax_err.plot(X, 1 - p(X)/func(X), f'-r', lw="3", label=f'n={points_number-1}')
        ax_err.set_title("Lagrange | Error")
        ax_err.set_xlabel("x")
        ax_err.set_ylabel("y")
        ax.legend()
        ax_err.legend()
        ax.grid()
        ax_err.grid()
        plt.show()
        return max(abs(1 - p(X)/func(X)))

    @staticmethod
    def test_by_spline(func, a=0, b=2, points_number=6, max_order=1):
        x = np.linspace(a, b, points_number)
        X = np.linspace(a, b, 1000)
        y = func(x)
        p = interp(x, y, max_order)
        fig, ax = plt.subplots()
        fig_err, ax_err = plt.subplots()

        ax.plot(X, func(X), '-r', label="func", lw="4")
        ax.plot(x, y, 'ob', ms="8", label="points")
        ax.plot(X, p(X), f'--g', lw="3", label=f'n={max_order}')
        ax.set_title("Spline | Graphics")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax_err.plot(X, 1 - p(X)/func(X), f'-r', lw="3", label=f'n={max_order}')
        ax_err.set_title("Spline | Error")
        ax_err.set_xlabel("x")
        ax_err.set_ylabel("y")
        ax.legend()
        ax_err.legend()
        ax.grid()
        ax_err.grid()
        plt.show()
        return max(abs(1 - p(X)/func(X)))


def main():
    a = 0
    b = 3.1415926
    t_x0_number = 5
    t_points_number = 7
    t_max_order = 3
    lg_points_number = 11
    sp_points_number = 6
    sp_max_order = 3

    xx = Symbol('xx')
    init_printing(use_unicode=True)
    print("Calc...")
    c, err = ApproximateErrorTester.test_by_taylor_series(
        func(xx), xx,
        a=a,
        b=b,
        max_order=t_max_order,
        points_number=t_points_number,
        x0_number=t_x0_number
    )
    print("=============================TAYLOR=============================")

    for e, i in zip(err, c):
        print(f'c={round(i, 3)}', end=": ")
        for er, j in zip(e, range(t_x0_number)):
            print(f'|n={j+1}:{er}', end="")
        print("|")
    print(
        f'min c={round(c[list(map(LA.norm, err)).index(min(list(map(LA.norm, err))))], 3)}'
        + f' of {len(c)} calc points'
    )
    # print("============================LAGRANGE============================")
    # err = ApproximateErrorTester.test_by_lagrange(
    #     func,
    #     a=a,
    #     b=b,
    #     points_number=lg_points_number
    # )
    # print(f'max err = {err}, n={lg_points_number-1}')

    # print("=============================SPLINE=============================")
    # err = ApproximateErrorTester.test_by_taylor_series(
    #     func,
    #     a=a,
    #     b=b,
    #     points_number=sp_points_number,
    #     max_order=sp_max_order
    # )
    # print(f'max err = {err}, n={sp_max_order}')
    # print("================================" * 2)


if __name__ == "__main__":
    main()