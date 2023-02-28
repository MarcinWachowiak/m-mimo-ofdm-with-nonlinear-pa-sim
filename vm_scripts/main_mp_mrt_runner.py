"""
Multiprocessing/parallel version of the:
simulate tha antenna array radiation pattern in the presence of precoding.
"""

import subprocess

if __name__ == '__main__':
    # %%
    # Crude multiprocessing
    n_ant_vec = [16, 32, 64, 128]
    channel_type_lst = ["los", "two_path", "rayleigh"]

    process_list = list()
    for n_ant_val in n_ant_vec:
        for channel_str in channel_type_lst:
            process_list.append(subprocess.Popen(
                ["main_beampatterns_plotting/main_mp_mrt_precoding_radiation_pattern.py", str(n_ant_val),
                 str(channel_str)]))
    for p_idx in range(len(process_list)):
        process_list[p_idx].wait()

    print("Finished processing!")
