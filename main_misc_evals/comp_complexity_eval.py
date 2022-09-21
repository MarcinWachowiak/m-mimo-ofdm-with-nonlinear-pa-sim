# %%
import numpy as np

# %%
# params
M = 64
N_U = 2048
N = 4096
K = 64
I = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
# standard rx:
std_add = 5 * N_U + 5 * ((N / 2) * np.log2(N)) + 2 * N * np.log2(N) + N_U * (3 * M)
std_mul = 3 * N_U + 3 * ((N / 2) * np.log2(N)) + N_U * (2 * M)
print(std_add)
print(std_mul)

# CNC rx:
cnc_add = 5 * N_U + 5 * ((N / 2) * np.log2(N)) + 2 * N * np.log2(N) + N_U * (3 * M) + I * (
        2 * (5 * ((N / 2) * np.log2(N)) + 2 * N * np.log2(N)) + 70 * N + 2 * N_U + N_U * (3 * M))
cnc_mul = 3 * N_U + 3 * ((N / 2) * np.log2(N)) + N_U * (2 * M) + I * (
        2 * (3 * ((N / 2) * np.log2(N))) + 5 * N + 2 * N_U + N_U * (2 * M))
print("cnc_add ", cnc_add)
print("cnc_mul ", cnc_mul)

# MCNC rx:
mcnc_add = 5 * N_U + 5 * ((N / 2) * np.log2(N)) + 2 * N * np.log2(N) + N_U * (3 * M) + I * (
        (K + 1) * (5 * ((N / 2) * np.log2(N)) + 2 * N * np.log2(N)) + K * (70 * N) + (2 * K + 1) * (5 * N_U) + (
        K - 1) + 2 * N_U + N_U * (3 * M))
mcnc_mul = 3 * N_U + 3 * ((N / 2) * np.log2(N)) + N_U * (2 * M) + I * (
        (K + 1) * (3 * ((N / 2) * np.log2(N))) + K * (5 * N) + (2 * K + 1) * 3 * N_U + N_U * (2 * M))
print("mcnc_add", mcnc_add)
print("mcnc_mul", mcnc_mul)


def human_format(num):
    return '%.2f' % (num / 1e6)
    # magnitude = 0
    # while abs(num) >= 1000:
    #     magnitude += 1
    #     num /= 1000.0
    # # add more suffixes if you need them
    # return '%.2f%s' % (num, ['', 'K', 'M', 'G', 'T', 'P'][magnitude])


for ite in I:
    print(r"%d & %s & %s & %s & %s \\ " % (
    ite, human_format(cnc_add[ite]), human_format(mcnc_add[ite]), human_format(cnc_mul[ite]),
    human_format(mcnc_mul[ite])))
    print("\hline")
