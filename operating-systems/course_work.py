variant = 195
V = 16
H = 12
q = 5
M = 10
K = [i for i in range(10)]
v = [6, 3, 2, 4, 3, 5, 7, 9, 4, 1]
h = [2, 4, 3, 1, 2, 0, 4, 1, 6, 3]
tau = [60, 90, 20, 10, 60, 30, 70, 30, 40, 20]
X = [195]
K = []
t_post = []
t_zag = []
t_zav_sjf = [396, 108, 59, 260, 387, 109, 497, 168, 422, 524]
t_zav_fifo = [417, 103, 59, 210, 397, 182, 513, 182, 438, 524]
T = []
W_sjf = []
W_fifo = []


def main():
    for i in range(1, 11):
        X.append((7 * X[i - 1] + 417) % 1000)
    for i in range(1, 11):
        K.append((X[i] // 7) % 10)
    t_post.append(K[0])
    for i in range(1, 10):
        t_post.append(t_post[i - 1] + K[i])
    for i in range(10):
        t_zag.append(5 * h[K[i]])
        T.append(t_zag[i] + tau[K[i]])
        W_sjf.append((t_zav_sjf[i] - t_post[i]) / T[i])
        W_fifo.append((t_zav_fifo[i] - t_post[i]) / T[i])
    print(T)
    print(t_post)
    print(W_sjf)
    print(sum(W_sjf) / len(W_sjf))


if __name__ == '__main__':
    main()
