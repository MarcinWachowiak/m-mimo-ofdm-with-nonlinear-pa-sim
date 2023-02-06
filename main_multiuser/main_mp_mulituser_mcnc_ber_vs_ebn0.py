# antenna array evaluation
# %%
import os
import sys

sys.path.append(os.getcwd())

from main_multiuser.multiuser_mp_prototype import multiuser_mcnc_parallel

import multiprocessing as mp
import numpy as np

if __name__ == '__main__':

    num_cores = 12
    precoding_str = 'mr'
    bits_sent_max = int(1e7 / num_cores)
    n_err_min = int(1e6 / num_cores)

    seed_rng = np.random.default_rng(2137)
    proc_seed_lst = seed_rng.integers(0, high=sys.maxsize, size=(num_cores, 2))



    processes = []
    for idx in range(num_cores):
        p = mp.Process(target=multiuser_mcnc_parallel, args=(idx, precoding_str, bits_sent_max, n_err_min, proc_seed_lst[idx]))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    print("Finished processing!")
