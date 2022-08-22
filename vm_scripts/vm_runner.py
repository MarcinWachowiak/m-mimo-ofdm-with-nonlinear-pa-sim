# VM simulation runner
import os
import sys
import time
from datetime import datetime

sys.path.append(os.getcwd())

filename_str_lst = [
    # "main_mp_clipping_noise_cancellation/main_mp_miso_cnc_ber_vs_ebn0.py",
    # "main_mp_clipping_noise_cancellation/main_mp_miso_cnc_ber_vs_ibo.py",
    # "main_mp_clipping_noise_cancellation/main_mp_miso_cnc_ber_vs_nant_vs_chan.py",

    # "main_mp_clipping_noise_cancellation/main_mp_miso_mcnc_ber_vs_ebn0.py",
    # "main_mp_clipping_noise_cancellation/main_mp_miso_cnc_ber_vs_nant_vs_chan.py"
    # "main_mp_clipping_noise_cancellation/main_mp_miso_mcnc_ber_vs_ibo.py"
    # 
    "main_mp_clipping_noise_cancellation/main_mp_miso_cnc_constant_ber_req_ebn0_vs_ibo.py",
    "main_mp_clipping_noise_cancellation/main_mp_miso_mcnc_constant_ber_req_ebn0_vs_ibo.py",
    "main_beampatterns_plotting/main_multiuser_mrt_precoding_radiation_pattern.py"

]

for idx, filename_str in enumerate(filename_str_lst):
    print("Running: %d/%d, %s" % (idx + 1, len(filename_str_lst), filename_str))

    start_time = time.time()
    print("### Start time: %s ###" % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    try:
        os.system("python3 %s" % filename_str)
    except:
        print("Runtime error occurred!")

    print("### Computation time: %f ###" % (time.time() - start_time))

print("Finished computation!")
