# VM simulation runner
import os, sys
import time
from datetime import datetime

sys.path.append(os.getcwd())
# "main_beampatterns_plotting/main_sdr_vs_ibo_vs_channel.py",
# "main_clipping_noise_cancellation/main_miso_cnc_ber_vs_ebn0.py"
# "main_clipping_noise_cancellation/main_miso_mcnc_ber_vs_ebn0.py",


filename_str_lst = ["main_clipping_noise_cancellation/main_miso_cnc_ber_vs_ibo.py",
                    "main_clipping_noise_cancellation/main_miso_mcnc_ber_vs_ibo.py",
                    "main_clipping_noise_cancellation/main_miso_cnc_constant_ber_req_ebn0_vs_ibo.py",
                    "main_clipping_noise_cancellation/main_miso_mcnc_constant_ber_req_ebn0_vs_ibo.py"
                    ]
for idx, filename_str in enumerate(filename_str_lst):
    print("Running: %d/%d, %s" %(idx+1, len(filename_str_lst), filename_str))

    start_time = time.time()
    print("--- Start time: %s ---" % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    # SDR VS IBO VS N_ANT VS CHANNEL
    try:
        os.system("python3 %s" % filename_str)
    except:
        print("Runtime error occurred!")

    print("--- Computation time: %f ---" % (time.time() - start_time))

print("Finished computation!")
