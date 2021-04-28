# -16y + 13y**2 + 2xy
import numpy as np
import scipy.integrate as si
import scipy.interpolate as interp
import matplotlib.pyplot as plt


def f(t, y):
    return -16 * y + 13 * y**2 + 2 * t * y


def gen_np_arr(arr: list):
    return np.array(arr, dtype=np.float64)


def Euler(f, n=10, h=0.1, t=0, y=1):
    for i in range(n):
        y += h * f(t, y)
        t += h
    return t, y


def NeyawnEuler(f, n=10, h=0.1, t=0, y=1) -> tuple:
    for i in range(n):
        yt = y + h * f(t, y)
        y = y + h * ((f(t, y) + f(t + h, yt)) / 2)
        t += h
    return t, y


def main() -> None:
    n = [5, 10, 20, 50, 100]
    etalon = si.solve_ivp(f, [0, 1], [1], t_eval=np.linspace(0, 1, 100))

    for i in range(len(n)):
        tspan = np.linspace(0, 1, n[i])
        sol = si.solve_ivp(f, [tspan[0], tspan[-1]], [1], t_eval=tspan)
        x = []
        ye = []
        yne = []
        for j in range(len(sol.t)):
            xxe, yye = Euler(f, n=j, h=1 / len(sol.t))
            xxne, yyne = NeyawnEuler(f, n=j, h=1 / len(sol.t))
            x.append(xxe)
            ye.append(yye)
            yne.append(yyne)
        fig, ax = plt.subplots()
        ax.plot(etalon.t, etalon.y[0], '-r', lw=2, label='solution')
        ax.plot(sol.t, sol.y[0], '--b', lw=3, label='RungeKutt')
        ax.plot(x, ye, '--g', lw=3, label='Euler')
        ax.plot(x, yne, '-.y', lw=3, label='NeyawnEuler')
        ax.set_title(f'ode, h={1/n[i]}')
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.legend()
        ax.grid()

        plt.show()


if __name__ == '__main__':
     main()
