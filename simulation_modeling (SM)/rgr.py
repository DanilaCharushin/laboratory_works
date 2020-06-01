from math import factorial as fact, log


def main():
    AFTER_COMMA = 10  # number of symbols after comma
    N = 2            # number of service channels
    m = 1            # queue length limit
    m_o = 522        # mathematical expectation of service time
    m_p = 103        # mathematical expectation of the time of arrival of applications
    lam = 1 / m_p    # input flow rate
    mu = 1 / m_o     # service flow rate

    print(f'lam = {round(lam, AFTER_COMMA)}')
    print(f'mu = {round(mu, AFTER_COMMA)}')
    a = lam / mu
    a_sum = a / N
    p0 = 1 / (1 + sum(a ** j / fact(j) for j in range(1, N + 1)) + a ** N * ((a_sum - a_sum ** (m + 1)) \
                                                                             / (fact(N) * (1 - a_sum))))
    print(f'p0 = {round(p0, AFTER_COMMA)}')
    p = lambda j: p0 * (a ** j / fact(j) if 1 <= j <= N else a ** j / (fact(N) * N ** (j - N)))
    p_zag = 1 - p0
    p_otk = p(N + m)
    p_pr = p0
    t_ob = (1 - p_otk) / mu
    N_z = a * (1 - p0 * a ** (N + m) / (N ** m * fact(N)))
    t_ozh = sum(i * p(N + i - 1) / (N * mu) for i in range(1, m + 1))
    n0 = sum(i * p(N + i) for i in range(1, m + 1))
    J = n0 + N_z
    t_s = t_ozh + t_ob
    for i in range(1, N + m + 1):
        print(f'p({i}) = {round(p(i), AFTER_COMMA)}')
    print(f'p_zag = {round(p_zag, AFTER_COMMA)}')
    print(f'p_otk = {round(p_otk, AFTER_COMMA)}')
    print(f'p_pr = {round(p_pr, AFTER_COMMA)}')
    print(f't_ob = {round(t_ob, AFTER_COMMA)}')
    print(f'N_z = {round(N_z, AFTER_COMMA)}')
    print(f't_ozh = {round(t_ozh, AFTER_COMMA)}')
    print(f'n0 = {round(n0, AFTER_COMMA)}')
    print(f'J = {round(J, AFTER_COMMA)}')
    print(f't_s = {round(t_s, AFTER_COMMA)}')


if __name__ == '__main__':
    main()
