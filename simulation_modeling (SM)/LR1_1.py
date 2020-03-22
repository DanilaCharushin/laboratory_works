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
print("┌─────┬────────────┬─────────────┬──────────────┐")
print("|  t  |    X(t)    |    dX(t)    |    dXm(t)    |")
print("├─────┼────────────┼─────────────┼──────────────┤")
avg = 0
for i in range(N):
    t = i * dt
    T.append(t)
    x.append(X(t))
    dx.append(dX(t))
    dxm.append(dXm(t))
    print("|" + str(round(t, 2)).center(5), end='')
    print("|" + str(round(X(t), 5)).center(12), end='')
    print("|" + str(round(dX(t), 5)).center(13), end='')
    print("|" + str(round(dXm(t), 5)).center(14) + "|")
    avg += abs(round(dX(t), 5) - round(dXm(t), 5))

avg /= N
print (f'avg = {round(avg, 5)}')

fig, ax = plt.subplots()
plt.title("DIFF. FILTER FUNCTION")
plt.xlabel("X")
plt.ylabel("Y")

ax.plot(T, x, "-r", label="X(t)")
ax.plot(T, dx, '-g', lw=5, label="dX(t)")
ax.plot(T, dxm, "-y", label="dXm(t)")
ax.grid(True)
ax.legend()
plt.show()
