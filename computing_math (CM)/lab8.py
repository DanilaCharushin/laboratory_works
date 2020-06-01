import numpy as np
import scipy.misc as sm
from scipy.optimize import minimize, approx_fprime
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

# MIN
def main():
    many_var()


def one_var():
    def f(x):
        return x ** 4 / np.log(x)

    def find_local(f, a, b, eps):
        x = np.linspace(a, b, 1000)
        plt.plot(x, f(x))
        # plt.plot(x, sm.derivative(f, x, dx=1e-6))
        plt.grid()

        xt = (a + b) / 2
        df = sm.derivative(f, xt, dx=1e-6)
        while abs(df) > eps:
            if df > 0:
                b = xt
            elif df < 0:
                a = xt
            xt = (a + b) / 2
            df = sm.derivative(f, xt, dx=1e-6)
        plt.vlines(xt, min(f(x)), max(f(x)))
        plt.show()
        print("local: (" + str(xt) + ";" + str(f(xt)) + ")")

    def find_global(f, a, b, eps, n=5):
        x = np.linspace(a, b, n)
        first = True
        while True:
            gl_min = f(x[0])
            if first:
                first = False
                gl_min_prev = max(f(x))

            for j in range(1, len(x) - 1):
                if f(x[j]) < gl_min:
                    gl_min = f(x[j])
                    x1 = x[j - 1]
                    x2 = x[j + 1]
            if abs(gl_min - gl_min_prev) > eps:
                x = np.linspace(x1, x2, n)
                gl_min_prev = gl_min
            else:
                ans = gl_min
                break
        print("global: (" + str((x1 + x2) / 2) + ";" + str(ans) + ")")

    a = 1.1
    b = 1.5
    eps = 0.00001
    find_local(f, a, b, eps)
    find_global(f, a, b, eps, 10)


def many_var():
    rz = lambda x: (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2
    h_1 = lambda x: (x[0] - 2 * x[1] + 2)
    h_2 = lambda x: (-x[0] - 2 * x[1] + 6)
    h_3 = lambda x: (-x[0] + 2 * x[1] + 2)

    xx = [0, 3]

    x0 = np.array(xx, dtype=np.float64)
    x_c = np.array(xx, dtype=np.float64)
    x_g = np.array(xx, dtype=np.float64)

    cons = ({'type': 'ineq', 'fun': h_1},
            {'type': 'ineq', 'fun': h_2},
            {'type': 'ineq', 'fun': h_3})
    x_m = minimize(rz, x0, constraints=cons).x
    print(f'scipy: {x_m}, {rz(x_m)=}')

    i = 1
    r = 1
    b = 0.2
    eps = 0.1
    while i < 1000:
        curr_func = lambda x: rz(x) + r * (1.0 / (h_1(x) ** 2 + h_2(x) ** 2 + h_3(x) ** 2))
        if curr_func(x_c) < eps:
            break
        x_c = minimize(curr_func, x_c).x
        i += 1
        r *= b
    print(f'barier: {x_c}, {rz(x_c)=}')

    step = 0.001
    eps = 1e-6
    while True:
        x_prev = x_g
        x_g = x_g - step * approx_fprime(x_g, rz, epsilon=eps)
        if np.linalg.norm(x_g - x_prev) < eps:
            break

    eps = 0.01
    X, Y = np.meshgrid(np.arange(-2, 2, eps), np.arange(-1, 3, eps))
    Z = (1 - X) ** 2 + 100 * (Y - X ** 2) ** 2
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # ax.plot_surface(X, Y, Z, cmap='twilight_shifted')
    ax.plot_surface(X, Y, np.log(Z), cmap='twilight_shifted')
    # ax.view_init(azim=120, elev=0)
    print(f'gradient: {x_g}, {rz(x_g)=}')


    plt.show()


if __name__ == '__main__':
    main()
