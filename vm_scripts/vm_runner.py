# VM simulation runner
import os
import sys
import time
import subprocess
from datetime import datetime

sys.path.append(os.getcwd())

filename_str_lst = [
    "main_mp_clipping_noise_cancellation\main_mp_miso_cnc_ber_vs_ibo.py",
    "main_mp_clipping_noise_cancellation\main_mp_miso_mcnc_ber_vs_ibo.py",
]

for idx, filename_str in enumerate(filename_str_lst):
    print("Running: %d/%d, %s" % (idx + 1, len(filename_str_lst), filename_str))

    start_time = time.time()
    print("### Start time: %s ###" % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    try:
        subprocess.call("python %s" % filename_str, shell=False)
    except:
        print("Runtime error occurred!")

    print("### Computation time: %f ###" % (time.time() - start_time))

print("Finished computation!")
