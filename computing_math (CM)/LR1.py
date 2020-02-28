from random import *
from math import *
import matplotlib.pyplot as plt

# x**3-x**2+2x+3=0

a = -5
b = 5
eps = 0.0001


def F(x):  # F(x)
    return x ** 3 - x ** 2 + 2 * x + 3


def dF(x):  # F'(x)
    return 3 * x ** 2 - 2 * x + 2


def ddF(x):  # F''(x)
    return 6 * x - 2


X = []
Y = []
dY = []
ddY = []
XN1 = []
XN2 = []
XN3 = []
N1 = []
N2 = []
N3 = []
x = a
while x < b:
    X.append(x)
    Y.append(F(x))
    dY.append(dF(x))
    ddY.append(ddF(x))
    x += eps

# считаем количество знаком после запятой-----------------------------------
n = int(1 / eps)
signs = 0
while n > 9:
    signs += 1
    n //= 10
# --------------------------------------------------------------------------

# метод половинного деления-------------------------------------------------
i = 0
while (b - a) > eps:
    d = (a + b) / 2
    XN1.append(d)
    N1.append(i)
    i += 1
    if F(a) * F(d) < 0:
        b = d
    else:
        a = d

x = round((a + b) / 2, signs + 1)
d_eps = round((b - a) / 2, signs + 1)
print(f'Метод половинного делений: x = {x}, ошибка = {d_eps}')
# ---------------------------------------------------------------------------


# метод хорд-----------------------------------------------------------------
xn = 0
xnn = 0
a = -2
b = 0

if F(b) * ddF(a) > 0:
    xn = a
    xnn = a
    i = 0
    while True:
        XN2.append(xn)
        N2.append(i)
        i += 1
        xn = xnn
        xnn = b - F(b) * (b - xn) / (F(b) - F(xn))
        if abs(xnn - xn) < eps:
            break
elif F(a) * ddF(b) > 0:
    xn = b
    xnn = b
    i = 0
    while True:
        XN2.append(xn)
        N2.append(i)
        i += 1
        xn = xnn
        xnn = a - F(a) * (xn - a) / (F(xn) - F(a))
        if abs(xnn - xn) < eps:
            break
else:
    print("Не могу решить уравнение методом хорд")

x = round(xnn, signs)
d_eps = round(abs(xnn - xn), signs + 1)
print(f'Метод хорд: x = {x}, ошибка = {d_eps}')
# ---------------------------------------------------------------------------


# метод касательных----------------------------------------------------------
xn = 0
xnn = 0
a = -2
b = 0

if F(b) * ddF(a) > 0:
    xn = b
    xnn = b
elif F(a) * ddF(b) > 0:
    xn = a
    xnn = a
else:
    print("Не могу решить уравнение методом касательных")

i = 0
while True:
    XN3.append(xn)
    N3.append(i)
    i += 1
    xn = xnn
    xnn = xn - F(xn) / dF(xn)
    if abs(xnn - xn) < eps:
        break

x = round(xnn, signs)
d_eps = round(abs(xnn - xn), signs + 1)
print(f'Метод касательных: x = {x}, ошибка = {d_eps}')
# ---------------------------------------------------------------------------

""" 
plt.plot(N1, XN1)
plt.plot(N2, XN2)
plt.plot(N3, XN3)
"""
plt.plot(X, Y)
plt.plot(X, dY)
plt.plot(X, ddY)
plt.show()
