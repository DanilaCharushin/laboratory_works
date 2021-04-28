import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as si
import scipy.misc as sc
import scipy.interpolate as interp


def main() -> None:
    task2_2()


def gen_np_arr(arr: list) -> np.ndarray:
    return np.array(arr, dtype=np.float64)


def make_round(arr: list, sign: int) -> list:
    temp_list = []
    for a in arr:
        temp_list.append(round(a, sign))
    return temp_list


def task1_1() -> None:
    x = gen_np_arr([1.0, 1.5, 2.0, 2.5])
    y = gen_np_arr([0.43, 0.28, 0.16, 0.76])
    c = gen_np_arr([1.45, 1.68, 2.1])
    fx3 = interp.interp1d(x, y, 3)
    fx2 = interp.interp1d(x, y, 2)
    fx1 = interp.interp1d(x, y, 1)
    h = 0.16

    d_left3 = []
    d_right3 = []
    d_cent3 = []
    dd_cent3 = []

    d_left2 = []
    d_right2 = []
    d_cent2 = []
    dd_cent2 = []

    d_left1 = []
    d_right1 = []
    d_cent1 = []
    dd_cent1 = []
    for i in range(len(c)):
        d_left3.append((fx3(c[i]) - fx3(c[i] - h)) / h)
        d_right3.append((fx3(c[i] + h) - fx3(c[i])) / h)
        d_cent3.append((fx3(c[i] + h) - (fx3(c[i] - h))) / 2 / h)
        dd_cent3.append((fx3(c[i] - h) - 2 * fx3(c[i]) + fx3(c[i] + h)) / h ** 2)

        d_left2.append((fx2(c[i]) - fx2(c[i] - h)) / h)
        d_right2.append((fx2(c[i] + h) - fx2(c[i])) / h)
        d_cent2.append((fx2(c[i] + h) - (fx2(c[i] - h))) / 2 / h)
        dd_cent2.append((fx2(c[i] - h) - 2 * fx2(c[i]) + fx2(c[i] + h)) / h ** 2)

        d_left1.append((fx1(c[i]) - fx1(c[i] - h)) / h)
        d_right1.append((fx1(c[i] + h) - fx1(c[i])) / h)
        d_cent1.append((fx1(c[i] + h) - (fx1(c[i] - h))) / 2 / h)
        dd_cent1.append((fx1(c[i] - h) - 2 * fx1(c[i]) + fx1(c[i] + h)) / h ** 2)

    d_left1 = make_round(d_left1, 5)
    d_right1 = make_round(d_right1, 5)
    d_cent1 = make_round(d_cent1, 5)
    dd_cent1 = make_round(dd_cent1, 5)

    d_left2 = make_round(d_left2, 5)
    d_right2 = make_round(d_right2, 5)
    d_cent2 = make_round(d_cent2, 5)
    dd_cent2 = make_round(dd_cent2, 5)

    d_left3 = make_round(d_left3, 5)
    d_right3 = make_round(d_right3, 5)
    d_cent3 = make_round(d_cent3, 5)
    dd_cent3 = make_round(dd_cent3, 5)

    print(f'{d_left1=}, {d_right1=}, {d_cent1=}, {dd_cent1=}')
    print(f'{d_left2=}, {d_right2=}, {d_cent2=}, {dd_cent2=}')
    print(f'{d_left3=}, {d_right3=}, {d_cent3=}, {dd_cent3=}')


def task1_2() -> None:
    def f(x) -> np.float64:
        return np.sin(x) * x

    def d_right(x, h) -> np.float64:
        return (f(x + h) - f(x)) / h

    def d_cent(x, h) -> np.float64:
        return (f(x + h) - f(x - h)) / 2 / h

    def dd_cent(x, h) -> np.float64:
        return (f(x - h) - 2 * f(x) + f(x + h)) / (h ** 2)

    a = 0
    b = 2 * 3.1415926
    h = [0.5, 0.1, 0.01]
    for i in range(len(h)):
        print(f'Error f\'(x)({h[i]=}, right) = {round(h[i] / 2, 8)}')
        print(f'Error f\'(x)({h[i]=}, central) = {round(h[i]**2 / 6, 8)}')
        print(f'Error f\'\'(x)({h[i]=}) = {round(h[i]**2 / 12, 8)}')

        fig, ax = plt.subplots()
        fig_err, ax_err = plt.subplots()
        x = np.linspace(a, b, 100)
        ax.plot(x, f(x), '-r', label='f(x)')
        ax.plot(x, sc.derivative(f, x, dx=1e-6), '-c', label='f\'(x)')
        ax.plot(x, sc.derivative(f, x, n=2, dx=1e-6), '-y', label='f\'\'(x)', lw=4)
        ax.plot(x, d_right(x, h[i]), '-g', label='f\'(x) (r)', lw=4)
        ax.plot(x, d_cent(x, h[i]), '-b', label='f\'(x) (c)', lw=1)
        ax.plot(x, dd_cent(x, h[i]), '-k', label='f\'\'(x) (c)', lw=1)
        ax.set_title(f'Derivations, {h[i]=}')
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.legend()
        ax.grid()

        ax_err.plot(x, x - x, '-r', label='0', lw=4)
        ax_err.plot(x, abs(sc.derivative(f, x, dx=1e-6) - d_right(x, h[i])), '--g', label='f\'(x)(r)', lw=2)
        ax_err.plot(x, abs(sc.derivative(f, x, dx=1e-6) - d_cent(x, h[i])), '--b', label='f\'(x)(c)', lw=2)
        ax_err.plot(x, abs(sc.derivative(f, x, n=2, dx=1e-6) - dd_cent(x, h[i])), '--y', label='f\'\'(x)', lw=2)
        ax_err.set_title(f'Derivations error, {h[i]=}')
        ax_err.set_xlabel("x")
        ax_err.set_ylabel("y")
        ax_err.legend()
        ax_err.grid()
    plt.show()


def task2_1() -> None:
    def f(x) -> np.float64:
        return 2 ** x

    def pr(f, a, b, n) -> np.float64:
        res = 0
        h = (b - a) / n
        for i in range(n):
            res += f(a + h * (i + 0.5))
        return res * h

    a = 0
    b = 5
    n = [1, 2, 4, 10, 50, 100]
    stand = (2 ** b - 2 ** a) / np.log(2)
    for i in range(len(n)):
        x = np.linspace(a, b, n[i])
        simp = si.simps(f(x), x)
        trapz = si.trapz(f(x), x)
        pryam = pr(f, a, b, n[i])

        print(f'{n[i]=}')
        print(f'{round(simp, 6)=}')
        print(f'{round(trapz, 6)=}')
        print(f'{round(pryam, 6)=}')
        print(f'{round(stand, 6)=}')


def task2_2() -> None:
    def f(x) -> np.float64:
        return x * np.sin(x ** 3)

    def my_simp(f, a, b, eps) -> np.float64:
        h = eps ** 0.85
        n = (b - a) / h
        n = int(n) + 1
        x = np.linspace(a, b, n)
        y = f(x)
        print(n)
        return h * (y[0] + y[-1] + 4 * (sum(y[i] for i in range(1, n, 2))) \
                    + 2 * (sum(y[i] for i in range(2, n - 1, 2)))) / 3

    a = 0
    b = 3.1415926 / 3
    eps = 1e-8
    sign = 8
    print("simpson: " + str(round(my_simp(f, a, b, eps), sign)))
    print("standart: " + str(round(si.quad(f, a, b)[0], sign)))


def task2_3() -> None:
    x = []
    y = []
    with open("integral.txt") as f:
        line = f.readline()
        while line:
            xx, yy = map(float, line.split())
            x.append(xx)
            y.append(yy)
            line = f.readline()

    for i in range(len(x) - 1):
        for j in range(i, len(x)):
            if x[i] > x[i + 1]:
                x[i], x[i + 1] = x[i + 1], x[i]
                y[i], y[i + 1] = y[i + 1], y[i]

    plt.plot(x, y)
    plt.plot(x, y, '.r', ms="10")
    plt.vlines(x[0], 0, y[0])
    plt.vlines(x[-1], 0, y[-1])
    plt.hlines(0, x[0], x[-1])
    plt.grid()
    plt.show()
    trapz = si.trapz(x, y)
    print(round(trapz, 8))


if __name__ == '__main__':
    main()
