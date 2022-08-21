import os

if __name__ == '__main__':
    # %%
    # Crude multiprocessing
    n_ant_vec = [16, 32, 64, 128]
    channel_type_lst = ["los", "two_path", "rayleigh"]

    for n_ant_val in n_ant_vec:
        for channel_str in channel_type_lst:
            os.system("python3 main_mp_mrt_precoding_radiation_pattern.py %s %s" % (n_ant_val, channel_str))

    print("Finished processing!")
