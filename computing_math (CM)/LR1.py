from scipy.optimize import root as ROOT
import matplotlib.pyplot as plt
import matplotlib.ticker as tic
import numpy as np

# x**3-x**2+2x+3=0
# simpy - символьные вычисления

A = -5
B = 5
ACCURACY = 0.0001


# F(x)
def func(x):
    return x ** 3 - x ** 2 + 2 * x + 3


# F'(x)
def d_func(x):
    return 3 * x ** 2 - 2 * x + 2


# F''(x)
def dd_func(x):
    return 6 * x - 2


X = np.linspace(A, B, int(1 / ACCURACY))
Y = func(X)
dY = d_func(X)
ddY = dd_func(X)

X_HALF_DIVISION = []
Y_HALF_DIVISION = []
X_CHORD = []
Y_CHORD = []
X_TANGENT = []
Y_TANGENT = []


def half_division_method(func, a, b, accuracy):
    global X_HALF_DIVISION
    global Y_HALF_DIVISION
    i = 0
    signs = str(accuracy).count('0')

    while (b - a) / 2 > accuracy:
        d = (a + b) / 2
        X_HALF_DIVISION.append(i)
        Y_HALF_DIVISION.append(d)
        i += 1
        if func(a) * func(d) < 0:
            b = d
        else:
            a = d

    root = round((a + b) / 2, signs)
    error = round((b - a) / 2, signs + 1)

    return {
        "root": root,
        "error": error,
    }


def chord_method(func, dd_func, a, b, accuracy):
    global X_CHORD
    global Y_CHORD
    i = 0
    signs = str(accuracy).count('0')
    xn = 0
    xnn = 0

    if func(b) * dd_func(a) > 0:
        xn = a
        xnn = a
        while True:
            Y_CHORD.append(xn)
            X_CHORD.append(i)
            i += 1
            xn = xnn
            xnn = b - func(b) * (b - xn) / (func(b) - func(xn))
            if abs(xnn - xn) < accuracy:
                break
    elif func(a) * dd_func(b) > 0:
        xn = b
        xnn = b
        while True:
            Y_CHORD.append(xn)
            X_CHORD.append(i)
            i += 1
            xn = xnn
            xnn = a - func(a) * (xn - a) / (func(xn) - func(a))
            if abs(xnn - xn) < accuracy:
                break
    else:
        print("Can't solve by chord method")

    root = round(xnn, signs)
    error = round(abs(xnn - xn), signs + 2)

    return {
        "root": root,
        "error": error,
    }


def tangent_method(func, d_func, dd_func, a, b, accuracy):
    global X_TANGENT
    global Y_TANGENT
    i = 0
    signs = str(accuracy).count('0')
    xn = 0
    xnn = 0

    if func(b) * dd_func(a) > 0:
        xn = b
        xnn = b
    elif func(a) * dd_func(b) > 0:
        xn = a
        xnn = a
    else:
        print("Can't solve by tangent method")

    while True:
        X_TANGENT.append(i)
        Y_TANGENT.append(xn)
        i += 1
        xn = xnn
        xnn = xn - func(xn) / d_func(xn)
        if abs(xnn - xn) < accuracy:
            break

    root = round(xnn, signs)
    error = round(abs(xnn - xn), signs + 1)

    return {
        "root": root,
        "error": error,
    }


# считаем количество знаком после запятой

print("=========================================")
print("METHODS:")

# solving by half division method
# ===========================================================================
solution_half_division = half_division_method(func, A, B, ACCURACY)
root = solution_half_division["root"]
error = solution_half_division["error"]
print(f'Half division: root = {root}, error = {error}')
# ===========================================================================

# solving by chord method
# ===========================================================================
solution_chord_method = chord_method(func, dd_func, -2, 0, ACCURACY)
root = solution_chord_method["root"]
error = solution_chord_method["error"]
print(f'Chord method: root = {root}, error = {error}')
# ===========================================================================

# solving by tangent method
# ===========================================================================
solution_tangent_method = tangent_method(func, d_func, dd_func, -2, 0, ACCURACY)
root = solution_tangent_method["root"]
error = solution_tangent_method["error"]
print(f'Tangent method: root = {root}, error = {error}')
# ===========================================================================

print("=========================================")
print("SCIPY.OPTIMIZE.ROOT:")
solution = ROOT(func, 0)
root = round(solution.x[0], str(ACCURACY).count('0'))
print(root)
print("=========================================")


fig, ax = plt.subplots()

plt.title("METHOD COMPARISON")
plt.xlabel("count of iterations")
plt.ylabel("inaccuracy")
ax.xaxis.set_major_locator(tic.MultipleLocator(1))
ax.plot(
    X_HALF_DIVISION, Y_HALF_DIVISION,
    'o-b', lw=1, mec='b', ms=3,
    label="half division method"
)
ax.plot(
    X_CHORD, Y_CHORD,
    'o-r', lw=1, mec='r', ms=3,
    label="chord method"
)
ax.plot(
    X_TANGENT, Y_TANGENT,
    'o-g', lw=1, mec='g', ms=3,
    label="tangent method"
)
ax.hlines(root, 0, len(X_HALF_DIVISION), label="root")
ax.grid(True)
ax.legend()
plt.show()
