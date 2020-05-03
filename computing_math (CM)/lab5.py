import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate as interp

e = 2.718281828
pi = 3.1415926
a = 2
b = 10
p = 10
law = "uniform"
# n_fixed = 30


def f(x):
    return np.sin(x) / np.log(x) # 2-10
    # return np.sin(x) / x # 0.3-10.3
    # return np.log(x)**np.sin(x) # 6-10
    # return np.sin(x) / (e**x+x) # 0-2.5
    # return np.sin(x**2) + np.log(x) # 1-3


class DistributionLawError(Exception):
    pass


class SmoothPointNumberError(Exception):
    pass


class GridData(object):
    @staticmethod
    def smooth(y, n=3):
        res = []
        if n == 3:
            if len(y) < 3:
                raise SmoothPointNumberError("Points number must be >= 3")
            else:
                res.append((5 * y[0] + 2 * y[1] - y[2]) / 6)
                for i in range(1, len(y) - 1):
                    res.append((y[i - 1] + y[i] + y[i + 1]) / 3)
                res.append((5 * y[-1] + 2 * y[-2] - y[-3]) / 6)
        elif n == 5:
            if len(y) < 5:
                raise SmoothPointNumberError("Points number must be >= 5")
            else:
                res.append((3 * y[0] + 2 * y[1] + y[2] - y[4]) / 5)
                res.append((4 * y[0] + 3 * y[1] + 2 * y[2] + y[3]) / 10)
                for i in range(2, len(y) - 2):
                    res.append((y[i - 2] + y[i - 1] + y[i] + y[i + 1] + y[i + 2]) / 5)
                res.append((4 * y[-1] + 3 * y[-2] + 2 * y[-3] + y[-4]) / 10)
                res.append((3 * y[-1] + 2 * y[-2] + y[-3] - y[-5]) / 5)
        elif n == 7:
            if len(y) < 7:
                raise SmoothPointNumberError("Points number must be >= 7")
            else:
                res.append((39 * y[0] + 8 * y[1] - 4 * (y[2] + y[3] - y[4]) + y[5] - 2 * y[6]) / 42)
                res.append((8 * y[0] + 19 * y[1] + 16 * y[2] + 6 * y[3] - 4 * y[4] - 7 * y[5] + 4 * y[6]) / 42)
                res.append((-4 * y[0] + 16 * y[1] + 19 * y[2] + 12 * y[3] + 2 * y[4] - 4 * y[5] + y[6]) / 42)
                for i in range(3, len(y) - 3):
                    res.append((7 * y[i] + 6 * (y[i + 1] + y[i - 1]) + 3 * (y[i + 2] + y[i - 2]) - 2 * (y[i + 3] + y[i - 3])) / 21)
                res.append((-4 * y[-1] + 16 * y[-2] + 19 * y[-3] + 12 * y[-4] + 2 * y[-5] - 4 * y[-6] + y[-7]) / 42)
                res.append((8 * y[-1] + 19 * y[-2] + 16 * y[-3] + 6 * y[-4] - 4 * y[-5] - 7 * y[-6] + 4 * y[-7]) / 42)
                res.append((39 * y[-1] + 8 * y[-2] - 4 * y[-3] - 4 * y[-4] + y[5] + 4 * y[-6] - 2 * y[-7]) / 42)
        else:
            raise SmoothPointNumberError("Unknown smooth point number. Available: 3, 5, 7")
        return np.array(res, dtype=np.float64)

    @staticmethod
    def noise(y, p, law="uniform"):
        eps = abs(y * p / 100)
        if law == "uniform":
            return np.random.uniform(y - eps, y + eps)
        elif law == "normal":
            return np.random.normal(y, eps / 3)
        elif law == "":
            return np.random.exponential()
        else:
            raise DistributionLawError("Unknown distribution type. Available: normal, uniform")


def main():
    n = [10]

    X = np.linspace(a, b, 1000)
    Y = f(X)

    for n_fixed in n:
        # =========================================================================
        # =================================PART 1==================================
        # =========================================================================
        x = np.linspace(a, b, n_fixed)
        y = f(x)

        tmp = []
        tmp.append(a)
        for i in range(1, n_fixed - 1):
            tmp.append(GridData.noise(x[i], 5, law=law))
        tmp.append(b)
        x_rand = np.sort(tmp)
        y_rand = f(x_rand)

        y_noisy = GridData.noise(y, p, law=law)
        y_rand_noisy = GridData.noise(y_rand, p, law=law)

        y_smooth3 = GridData.smooth(y_noisy, 3)
        y_smooth5 = GridData.smooth(y_noisy, 5)
        y_smooth7 = GridData.smooth(y_noisy, 7)
        y_rand_smooth3 = GridData.smooth(y_rand_noisy, 3)
        y_rand_smooth5 = GridData.smooth(y_rand_noisy, 5)
        y_rand_smooth7 = GridData.smooth(y_rand_noisy, 7)

        err_noise = y - y_noisy
        err_smooth3 = y - y_smooth3
        err_smooth5 = y - y_smooth5
        err_smooth7 = y - y_smooth7
        err_rand_noise = y - y_rand_noisy
        err_rand_smooth3 = y - y_rand_smooth3
        err_rand_smooth5 = y - y_rand_smooth5
        err_rand_smooth7 = y - y_rand_smooth7

        err_norm = [
            {"norm": np.linalg.norm(err_noise), "type": "noise", "value": y_noisy},
            {"norm": np.linalg.norm(err_smooth3), "type": "smooth3", "value": y_smooth3},
            {"norm": np.linalg.norm(err_smooth5), "type": "smooth5", "value": y_smooth5},
            {"norm": np.linalg.norm(err_smooth7), "type": "smooth7", "value": y_smooth7},
            {"norm": np.linalg.norm(err_rand_noise), "type": "rand_noise", "value": y_rand_noisy},
            {"norm": np.linalg.norm(err_rand_smooth3), "type": "rand_smooth3", "value": y_rand_smooth3},
            {"norm": np.linalg.norm(err_rand_smooth5), "type": "rand_smooth5", "value": y_rand_smooth5},
            {"norm": np.linalg.norm(err_rand_smooth7), "type": "rand_smooth7", "value": y_rand_smooth7}
        ]
        print("======================================")
        print("==============ERROR NORM==============")
        print(f'n={n_fixed}, p={p}, law={law}')
        print("--------------------------------------")
        print(f'       err_noise: {round(err_norm[0]["norm"], 5)}')
        print(f'     err_smooth3: {round(err_norm[1]["norm"], 5)}')
        print(f'     err_smooth5: {round(err_norm[2]["norm"], 5)}')
        print(f'     err_smooth7: {round(err_norm[3]["norm"], 5)}')
        print(f'  err_rand_noise: {round(err_norm[4]["norm"], 5)}')
        print(f'err_rand_smooth3: {round(err_norm[5]["norm"], 5)}')
        print(f'err_rand_smooth5: {round(err_norm[6]["norm"], 5)}')
        print(f'err_rand_smooth7: {round(err_norm[7]["norm"], 5)}')
        print("--------------------------------------")
        data = err_norm[0]
        for i in range(4):
            if err_norm[i]["norm"] < data["norm"]:
                data = err_norm[i]
        data_rand = err_norm[4]
        for i in range(4, 8):
            if err_norm[i]["norm"] < data_rand["norm"]:
                data_rand = err_norm[i]

        print(f'     min: {data["type"]} ({round(data["norm"], 5)})')
        print(f'min_rand: {data_rand["type"]} ({round(data_rand["norm"], 5)})')

        # ===============================
        y_data = data["value"]
        y_data_rand = data_rand["value"]
        # ===============================

        fig, ax = plt.subplots()
        fig_err, ax_err = plt.subplots()
        fig_rand, ax_rand = plt.subplots()
        fig_rand_err, ax_rand_err = plt.subplots()

        ax.plot(X, Y, '-r', lw="3", label="func")
        ax.plot(x, y, 'ob', ms="8", label="points")
        ax.plot(x, y_noisy, 'oc', ms='6', label="noisy")
        ax.plot(x, y_noisy, '-c', lw='1')
        ax.plot(x, y_smooth3, 'oy', ms='6', label="smooth3")
        ax.plot(x, y_smooth3, '-y', lw='1')
        ax.plot(x, y_smooth5, 'og', ms='4', label="smooth5")
        ax.plot(x, y_smooth5, '-g', lw='1')
        ax.plot(x, y_smooth7, 'ok', ms='3', label="smooth7")
        ax.plot(x, y_smooth7, '-k', lw='1')
        ax.set_title(f'func, dx=const, n={n_fixed}, p={p}, law={law}')
        ax.set_xlabel("x")
        ax.set_ylabel("y = sin(x) / ln(x)")
        ax.legend()
        ax.grid()

        ax_err.axhline(0, c='r', label="func")
        ax_err.plot(x, err_noise, 'c', lw='2', label="noise")
        ax_err.plot(x, err_smooth3, 'y', lw='2', label="smooth3")
        ax_err.plot(x, err_smooth5, 'g', lw='2', label="smooth5")
        ax_err.plot(x, err_smooth7, 'k', lw='2', label="smooth7")
        ax_err.set_title(f'func err, dx=const, n={n_fixed}, p={p}, law={law}')
        ax_err.set_xlabel("x")
        ax_err.set_ylabel("y - y'")
        ax_err.legend()
        ax_err.grid()


        ax_rand.plot(X, Y, '-r', lw="3", label="func")
        ax_rand.plot(x_rand, y_rand, 'ob', ms="8", label="points")
        ax_rand.plot(x_rand, y_rand_noisy, 'oc', ms='6', label="noisy")
        ax_rand.plot(x_rand, y_rand_noisy, '-c', lw='1')
        ax_rand.plot(x_rand, y_rand_smooth3, 'oy', ms='6', label="smooth3")
        ax_rand.plot(x_rand, y_rand_smooth3, '-y', lw='1')
        ax_rand.plot(x_rand, y_rand_smooth5, 'og', ms='4', label="smooth5")
        ax_rand.plot(x_rand, y_rand_smooth5, '-g', lw='1')
        ax_rand.plot(x_rand, y_rand_smooth7, 'ok', ms='3', label="smooth7")
        ax_rand.plot(x_rand, y_rand_smooth7, '-k', lw='1')
        ax_rand.set_title(f'func, dx=rand, n={n_fixed}, p={p}, law={law}')
        ax_rand.set_xlabel("x")
        ax_rand.set_ylabel("y = sin(x) / ln(x)")
        ax_rand.legend()
        ax_rand.grid()

        ax_rand_err.axhline(0, c='r', label="func")
        ax_rand_err.plot(x_rand, err_rand_noise, 'c', lw='2', label="noise")
        ax_rand_err.plot(x_rand, err_rand_smooth3, 'y', lw='2', label="smooth3")
        ax_rand_err.plot(x_rand, err_rand_smooth5, 'g', lw='2', label="smooth5")
        ax_rand_err.plot(x_rand, err_rand_smooth7, 'k', lw='2', label="smooth7")
        ax_rand_err.set_title(f'func err, dx=rand, n={n_fixed}, p={p}, law={law}')
        ax_rand_err.set_xlabel("x")
        ax_rand_err.set_ylabel("y - y'")
        ax_rand_err.legend()
        ax_rand_err.grid()

        # =========================================================================
        # =================================PART 2==================================
        # =========================================================================
        lg = interp.lagrange(x, y_data)
        lin = interp.interp1d(x, y_data, kind="linear")
        quad = interp.interp1d(x, y_data, kind="quadratic")
        cub = interp.interp1d(x, y_data, kind="cubic")
        lg_rand = interp.lagrange(x_rand, y_data_rand)
        lin_rand = interp.interp1d(x_rand, y_data_rand, kind="linear", )
        quad_rand = interp.interp1d(x_rand, y_data_rand, kind="quadratic")
        cub_rand = interp.interp1d(x_rand, y_data_rand, kind="cubic")

        Y_lg = lg(X)
        Y_lin = lin(X)
        Y_quad = quad(X)
        Y_cub = cub(X)
        Y_lg_rand = lg_rand(X)
        Y_lin_rand = lin_rand(X)
        Y_quad_rand = quad_rand(X)
        Y_cub_rand = cub_rand(X)

        lg_err = Y - Y_lg
        lin_err = Y - Y_lin
        quad_err = Y - Y_quad
        cub_err = Y - Y_cub
        lg_err_rand = Y - Y_lg_rand
        lin_err_rand = Y - Y_lg_rand
        quad_err_rand = Y - Y_quad_rand
        cub_err_rand = Y - Y_cub_rand

        print("===========================================")
        print("============INTERPOLATION ERROR============")

        err_norm = [
            {"norm": np.linalg.norm(lg_err), "type": "lg"},
            {"norm": np.linalg.norm(lin_err), "type": "lin"},
            {"norm": np.linalg.norm(quad_err), "type": "quad"},
            {"norm": np.linalg.norm(cub_err), "type": "cub"},
            {"norm": np.linalg.norm(lg_err_rand), "type": "rand_lg"},
            {"norm": np.linalg.norm(lin_err_rand), "type": "rand_lin"},
            {"norm": np.linalg.norm(quad_err_rand), "type": "rand_quad"},
            {"norm": np.linalg.norm(cub_err_rand), "type": "rand_cub"}
        ]
        print(f'n={n_fixed}, p={p}, law={law}')
        print("--------------------------------------")
        print(f'       lg_err: {round(err_norm[0]["norm"], 5)}')
        print(f'      lin_err: {round(err_norm[1]["norm"], 5)}')
        print(f'     quad_err: {round(err_norm[2]["norm"], 5)}')
        print(f'      cub_err: {round(err_norm[3]["norm"], 5)}')
        print(f'  lg_err_rand: {round(err_norm[4]["norm"], 5)}')
        print(f' lin_err_rand: {round(err_norm[5]["norm"], 5)}')
        print(f'quad_err_rand: {round(err_norm[6]["norm"], 5)}')
        print(f' cub_err_rand: {round(err_norm[7]["norm"], 5)}')
        print("--------------------------------------")
        data = err_norm[0]
        for i in range(4):
            if err_norm[i]["norm"] < data["norm"]:
                data = err_norm[i]
        data_rand = err_norm[4]
        for i in range(4, 8):
            if err_norm[i]["norm"] < data_rand["norm"]:
                data_rand = err_norm[i]

        print(f'     min: {data["type"]} ({round(data["norm"], 5)})')
        print(f'min_rand: {data_rand["type"]} ({round(data_rand["norm"], 5)})')

        h = 1

        fig_interp, ax_interp = plt.subplots()
        fig_err_interp, ax_err_interp = plt.subplots()
        fig_rand_interp, ax_rand_interp = plt.subplots()
        fig_rand_err_interp, ax_rand_err_interp = plt.subplots()

        ax_interp.plot(X, Y, '-r', lw="4", label="func")
        ax_interp.plot(x, y, 'ob', ms="8", label="points")
        ax_interp.plot(x, y_data, 'ok', ms="5", label="data")
        ax_interp.plot(X, Y_lg, '--c', lw='2', label="lg")
        ax_interp.plot(X, Y_lin, '--y', lw='2', label="lin")
        ax_interp.plot(X, Y_quad, '--b', lw='2', label="quad")
        ax_interp.plot(X, Y_cub, '--g', lw='2', label="cub")
        ax_interp.set_title(f'func interp, dx=const, n={n_fixed}, p={p}, law={law}')
        ax_interp.set_xlabel("x")
        ax_interp.set_ylabel("y = sin(x) / ln(x)")
        ax_interp.legend()
        ax_interp.grid()

        ax_err_interp.axhline(0, c='r', label="func")
        ax_err_interp.plot(X, lg_err, 'c', lw='2', label="lg")
        ax_err_interp.plot(X, lin_err, 'y', lw='2', label="ling")
        ax_err_interp.plot(X, quad_err, 'b', lw='2', label="quad")
        ax_err_interp.plot(X, cub_err, 'g', lw='2', label="cub")
        ax_err_interp.set_title(f'interp err, dx=const, n={n_fixed}, p={p}, law={law}')
        ax_err_interp.set_xlabel("x")
        ax_err_interp.set_ylabel("y - interp(x)")
        ax_err_interp.legend()
        ax_err_interp.grid()

        ax_rand_interp.plot(X, Y, '-r', lw="4", label="func")
        ax_rand_interp.plot(x_rand, y_rand, 'ob', ms="8", label="points")
        ax_rand_interp.plot(x_rand, y_data_rand, 'ok', ms="5", label="data")
        ax_rand_interp.plot(X, Y_lg_rand, '--c', lw='2', label="lg_rand")
        ax_rand_interp.plot(X, Y_lin_rand, '--y', lw='2', label="lin_rand")
        ax_rand_interp.plot(X, Y_quad_rand, '--b', lw='2', label="quad_rand")
        ax_rand_interp.plot(X, Y_cub_rand, '--g', lw='2', label="cub_rand")
        ax_rand_interp.set_title(f'func interp, dx=rand, n={n_fixed}, p={p}, law={law}')
        ax_rand_interp.set_xlabel("x")
        ax_rand_interp.set_ylabel("y = sin(x) / ln(x)")
        ax_rand_interp.legend()
        ax_rand_interp.grid()
        ax_rand_interp.set_ylim(ymin=min(Y) - h, ymax=max(Y) + h)

        ax_rand_err_interp.axhline(0, c='r', label="func")
        ax_rand_err_interp.plot(X, lg_err_rand, 'c', lw='2', label="lg")
        ax_rand_err_interp.plot(X, lin_err_rand, 'y', lw='2', label="lin")
        ax_rand_err_interp.plot(X, quad_err_rand, 'b', lw='2', label="quad")
        ax_rand_err_interp.plot(X, cub_err_rand, 'g', lw='2', label="cub")
        ax_rand_err_interp.set_title(f'interp err, dx=rand, n={n_fixed}, p={p}, law={law}')
        ax_rand_err_interp.set_xlabel("x")
        ax_rand_err_interp.set_ylabel("y - interp(x)")
        ax_rand_err_interp.legend()
        ax_rand_err_interp.grid()
        ax_rand_err_interp.set_ylim(ymin=min(Y) - h, ymax=max(Y) + h)

        print("===========================================")
        print("============APPROXIMATION ERROR============")

        p2 = np.poly1d(np.polyfit(x, y_data, 2))
        p3 = np.poly1d(np.polyfit(x, y_data, 3))
        p4 = np.poly1d(np.polyfit(x, y_data, 4))
        p5 = np.poly1d(np.polyfit(x, y_data, 5))
        p10 = np.poly1d(np.polyfit(x, y_data, 10))
        p2_rand = np.poly1d(np.polyfit(x_rand, y_data_rand, 2))
        p3_rand = np.poly1d(np.polyfit(x_rand, y_data_rand, 3))
        p4_rand = np.poly1d(np.polyfit(x_rand, y_data_rand, 4))
        p5_rand = np.poly1d(np.polyfit(x_rand, y_data_rand, 5))
        p10_rand = np.poly1d(np.polyfit(x_rand, y_data_rand, 10))
        A = np.vstack([x, np.ones(len(x))]).T
        m, c = np.linalg.lstsq(A, y_data, rcond=None)[0]
        A_rand = np.vstack([x_rand, np.ones(len(x_rand))]).T
        m_rand, c_rand = np.linalg.lstsq(A_rand, y_data_rand, rcond=None)[0]

        Y_p2 = p2(X)
        Y_p3 = p3(X)
        Y_p4 = p4(X)
        Y_p5 = p5(X)
        Y_p10 = p10(X)
        Y_lstsq = m*x + c
        Y_p2_rand = p2_rand(X)
        Y_p3_rand = p3_rand(X)
        Y_p4_rand = p4_rand(X)
        Y_p5_rand = p5_rand(X)
        Y_p10_rand = p10_rand(X)
        Y_lstsq_rand = m_rand*x_rand + c_rand

        p2_err = Y - Y_p2
        p3_err = Y - Y_p3
        p4_err = Y - Y_p4
        p5_err = Y - Y_p5
        p10_err = Y - Y_p10
        lstsq_err = y - Y_lstsq
        p2_err_rand = Y - Y_p2_rand
        p3_err_rand = Y - Y_p3_rand
        p4_err_rand = Y - Y_p4_rand
        p5_err_rand = Y - Y_p5_rand
        p10_err_rand = Y - Y_p10_rand
        lstsq_err_rand = y - Y_lstsq_rand

        fig_approx, ax_approx = plt.subplots()
        fig_err_approx, ax_err_approx = plt.subplots()
        fig_rand_approx, ax_rand_approx = plt.subplots()
        fig_rand_err_approx, ax_rand_err_approx = plt.subplots()

        ax_approx.plot(X, Y, '-r', lw="4", label="func")
        ax_approx.plot(x, y, 'ob', ms="8", label="points")
        ax_approx.plot(x, y_data, 'ok', ms="5", label="data")
        ax_approx.plot(X, Y_p2, '--b', lw='2', label="p2")
        ax_approx.plot(X, Y_p3, '--g', lw='2', label="p3")
        ax_approx.plot(X, Y_p4, '--k', lw='2', label="p4")
        ax_approx.plot(X, Y_p5, '--c', lw='2', label="p5")
        ax_approx.plot(X, Y_p10, '--m', lw='2', label="p10")
        ax_approx.plot(x, Y_lstsq, '--', c='#ff7f0e', lw='2', label="lstsq")
        ax_approx.set_title(f'func approx, dx=const, n={n_fixed}, p={p}, law={law}')
        ax_approx.set_xlabel("x")
        ax_approx.set_ylabel("y = sin(x) / ln(x)")
        ax_approx.legend()
        ax_approx.grid()

        ax_err_approx.axhline(0, c='r', label="func")
        ax_err_approx.plot(X, p2_err, 'b', lw='2', label="p2")
        ax_err_approx.plot(X, p3_err, 'g', lw='2', label="p3")
        ax_err_approx.plot(X, p4_err, 'k', lw='2', label="p4")
        ax_err_approx.plot(X, p5_err, 'c', lw='2', label="p5")
        ax_err_approx.plot(X, p10_err, 'm', lw='2', label="p10")
        ax_err_approx.plot(x, lstsq_err, c='#ff7f0e', lw='2', label="lstsq")
        ax_err_approx.set_title(f'approx err, dx=const, n={n_fixed}, p={p}, law={law}')
        ax_err_approx.set_xlabel("x")
        ax_err_approx.set_ylabel("y - approx(x)")
        ax_err_approx.legend()
        ax_err_approx.grid()

        ax_rand_approx.plot(X, Y, '-r', lw="4", label="func")
        ax_rand_approx.plot(x_rand, y_rand, 'ob', ms="8", label="points")
        ax_rand_approx.plot(x_rand, y_data_rand, 'ok', ms="5", label="data")
        ax_rand_approx.plot(X, Y_p2_rand, '--b', lw='2', label="p2")
        ax_rand_approx.plot(X, Y_p3_rand, '--g', lw='2', label="p3")
        ax_rand_approx.plot(X, Y_p4_rand, '--k', lw='2', label="p4")
        ax_rand_approx.plot(X, Y_p5_rand, '--c', lw='2', label="p5")
        ax_rand_approx.plot(X, Y_p10_rand, '--m', lw='2', label="p10")
        ax_rand_approx.plot(x, Y_lstsq_rand, '--', c='#ff7f0e', lw='2', label="lstsq")
        ax_rand_approx.set_title(f'func approx, dx=rand, n={n_fixed}, p={p}, law={law}')
        ax_rand_approx.set_xlabel("x")
        ax_rand_approx.set_ylabel("y = sin(x) / ln(x)")
        ax_rand_approx.legend()
        ax_rand_approx.grid()
        ax_rand_approx.set_ylim(ymin=min(Y) - h, ymax=max(Y) + h)

        ax_rand_err_approx.axhline(0, c='r', label="func")
        ax_rand_err_approx.plot(X, p2_err_rand, 'b', lw='2', label="p2")
        ax_rand_err_approx.plot(X, p3_err_rand, 'g', lw='2', label="p3")
        ax_rand_err_approx.plot(X, p4_err_rand, 'k', lw='2', label="p4")
        ax_rand_err_approx.plot(X, p5_err_rand, 'c', lw='2', label="p5")
        ax_rand_err_approx.plot(X, p10_err_rand, 'm', lw='2', label="p10")
        ax_rand_err_approx.plot(x, lstsq_err_rand, c='#ff7f0e', lw='2', label="lstsq")
        ax_rand_err_approx.set_title(f'approx err, dx=rand, n={n_fixed}, p={p}, law={law}')
        ax_rand_err_approx.set_xlabel("x")
        ax_rand_err_approx.set_ylabel("y - approx(x)")
        ax_rand_err_approx.legend()
        ax_rand_err_approx.grid()
        ax_rand_err_approx.set_ylim(ymin=min(Y) - h, ymax=max(Y) + h)

        plt.show()


if __name__ == "__main__":
    main()
