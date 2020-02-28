from math import sin, cos
import matplotlib.pyplot as plt

# ЗАДАНИЕ:
# X(t): sin^2(t) + cos(t)
# X'(t): 2*sin(t)*cos(t) - sin(t) = sin(t) * (2*cos(t) - 1)
# N: 1000


N = 1000
dt = 0.01


# X(t)
def X(t):
    return sin(t) ** 2 + cos(t)


# X'(t)
def dX(t):
    return sin(t) * (2*cos(t) - 1)


# X'(t) методом dt: X'(t) = (X(t + dt) - X(t)) / dt
def dXm(t):
    global dt
    return (X(t + dt) - X(t)) / dt


T = []
x = []
dx = []
dxm = []
for i in range(N):
    t = i * dt
    T.append(t)
    x.append(X(t))
    dx.append(dX(t))
    dxm.append(dXm(t))

plt.plot(T, x)
plt.plot(T, dx)
plt.plot(T, dxm)
plt.show()