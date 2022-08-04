import copy

import numpy as np

import channel
import corrector
import utilities


class Link():
    def __init__(self, mod_obj, array_obj, std_rx_obj, chan_obj, noise_obj, rx_loc_var, n_err_min,
                 bits_sent_max, is_mcnc=False):
        self.my_mod = copy.deepcopy(mod_obj)
        self.my_array = copy.deepcopy(array_obj)
        self.my_standard_rx = copy.deepcopy(std_rx_obj)

        self.rx_loc_x = self.my_standard_rx.cord_x
        self.rx_loc_y = self.my_standard_rx.cord_x
        self.my_miso_chan = copy.deepcopy(chan_obj)

        self.my_noise = copy.deepcopy(noise_obj)

        if is_mcnc:
            self.my_cnc_rx = corrector.McncReceiver(self.my_array, self.my_miso_chan)
        else:
            self.my_cnc_rx = corrector.CncReceiver(copy.deepcopy(array_obj.base_transceiver.modem),
                                                   copy.deepcopy(array_obj.base_transceiver.impairment))

        self.my_noise.rng_gen = np.random.default_rng(0)
        self.loc_rng = np.random.default_rng(1)
        self.bit_rng = np.random.default_rng(2)
        self.rx_loc_var = rx_loc_var

        self.n_ant_val = len(self.my_array.array_elements)
        self.n_bits_per_ofdm_sym = self.my_mod.n_bits_per_ofdm_sym
        self.n_sub_carr = self.my_mod.n_sub_carr
        self.ibo_val_db = self.my_array.array_elements[0].impairment.ibo_db
        self.n_err_min = n_err_min
        self.bits_sent_max = bits_sent_max

        self.set_precoding_and_recalculate_agc()

    def simulate(self, incl_clean_run, reroll_chan, cnc_n_iter_lst, seed_arr, n_err_shared_arr,
                 n_bits_sent_shared_arr):
        self.bit_rng = np.random.default_rng(seed_arr[0])
        self.my_noise.rng_gen = np.random.default_rng(seed_arr[1])
        self.loc_rng = np.random.default_rng(seed_arr[2])

        res_idx = 0
        if incl_clean_run:
            res_idx = 1
            # clean RX run
            while True:
                if np.logical_and((n_err_shared_arr[0] < self.n_err_min),
                                  (n_bits_sent_shared_arr[0] < self.bits_sent_max)):

                    if reroll_chan:
                        if isinstance(self.my_miso_chan, channel.MisoLosFd) or isinstance(self.my_miso_chan,
                                                                                          channel.MisoTwoPathFd):
                            # reroll location
                            self.my_standard_rx.set_position(
                                cord_x=self.rx_loc_x + self.loc_rng.uniform(low=-self.rx_loc_var / 2.0,
                                                                            high=self.rx_loc_var / 2.0),
                                cord_y=self.rx_loc_x + self.loc_rng.uniform(low=-self.rx_loc_var / 2.0,
                                                                            high=self.rx_loc_var / 2.0),
                                cord_z=self.my_standard_rx.cord_z)
                            self.my_miso_chan.calc_channel_mat(tx_transceivers=self.my_array.array_elements,
                                                               rx_transceiver=self.my_standard_rx,
                                                               skip_attenuation=False)
                        else:
                            self.my_miso_chan.reroll_channel_coeffs()

                        self.set_precoding_and_recalculate_agc()

                    tx_bits = self.bit_rng.choice((0, 1), self.n_bits_per_ofdm_sym)
                    clean_ofdm_symbol = self.my_array.transmit(tx_bits, out_domain_fd=True, return_both=False,
                                                               skip_dist=True)
                    clean_rx_ofdm_symbol = self.my_miso_chan.propagate(in_sig_mat=clean_ofdm_symbol)
                    clean_rx_ofdm_symbol = self.my_noise.process(clean_rx_ofdm_symbol,
                                                                 avg_sample_pow=self.my_mod.avg_symbol_power * self.hk_vk_noise_scaler,
                                                                 disp_data=False)
                    clean_rx_ofdm_symbol = np.divide(clean_rx_ofdm_symbol, self.hk_vk_agc_nfft)
                    clean_rx_ofdm_symbol = utilities.to_time_domain(clean_rx_ofdm_symbol)
                    clean_rx_ofdm_symbol = np.concatenate(
                        (clean_rx_ofdm_symbol[-self.my_mod.cp_len:], clean_rx_ofdm_symbol))
                    rx_bits = self.my_standard_rx.receive(clean_rx_ofdm_symbol)

                    n_bit_err = utilities.count_mismatched_bits(tx_bits, rx_bits)
                    n_err_shared_arr[0] += n_bit_err
                    n_bits_sent_shared_arr[0] += self.my_mod.n_bits_per_ofdm_sym
                else:
                    break
            # distorted RX run
        n_err_shared_arr_np = np.frombuffer(n_err_shared_arr.get_obj())
        n_bits_sent_shared_arr_np = np.frombuffer(n_bits_sent_shared_arr.get_obj())

        while True:
            ite_use_flags = np.logical_and((n_err_shared_arr_np[res_idx:] < self.n_err_min),
                                           (n_bits_sent_shared_arr_np[res_idx:] < self.bits_sent_max))

            if ite_use_flags.any() == True:
                curr_ite_lst = cnc_n_iter_lst[ite_use_flags]
            else:
                break

            # for direct visibility channel and CNC algorithm channel impact must be averaged
            if reroll_chan:
                if isinstance(self.my_miso_chan, channel.MisoLosFd) or isinstance(self.my_miso_chan,
                                                                                  channel.MisoTwoPathFd):
                    # reroll location
                    self.my_standard_rx.set_position(
                        cord_x=self.rx_loc_x + self.loc_rng.uniform(low=-self.rx_loc_var / 2.0,
                                                                    high=self.rx_loc_var / 2.0),
                        cord_y=self.rx_loc_x + self.loc_rng.uniform(low=-self.rx_loc_var / 2.0,
                                                                    high=self.rx_loc_var / 2.0),
                        cord_z=self.my_standard_rx.cord_z)
                    self.my_miso_chan.calc_channel_mat(tx_transceivers=self.my_array.array_elements,
                                                       rx_transceiver=self.my_standard_rx,
                                                       skip_attenuation=False)
                else:
                    self.my_miso_chan.reroll_channel_coeffs()

                self.set_precoding_and_recalculate_agc()

            tx_bits = self.bit_rng.choice((0, 1), self.n_bits_per_ofdm_sym)
            tx_ofdm_symbol = self.my_array.transmit(tx_bits, out_domain_fd=True, skip_dist=False)
            rx_ofdm_symbol = self.my_miso_chan.propagate(in_sig_mat=tx_ofdm_symbol)
            rx_ofdm_symbol = self.my_noise.process(rx_ofdm_symbol,
                                                   avg_sample_pow=self.my_mod.avg_symbol_power * self.ak_hk_vk_noise_scaler)

            rx_ofdm_symbol = np.divide(rx_ofdm_symbol, self.ak_hk_vk_agc_nfft)
            rx_bits_per_iter_lst = self.my_cnc_rx.receive(n_iters_lst=curr_ite_lst, in_sig_fd=rx_ofdm_symbol)

            ber_idx = np.array(list(range(len(cnc_n_iter_lst))))
            act_ber_idx = ber_idx[ite_use_flags] + res_idx
            for idx in range(len(rx_bits_per_iter_lst)):
                n_bit_err = utilities.count_mismatched_bits(tx_bits, rx_bits_per_iter_lst[idx])
                n_err_shared_arr[act_ber_idx[idx]] += n_bit_err
                n_bits_sent_shared_arr[act_ber_idx[idx]] += self.n_bits_per_ofdm_sym

    def update_distortion(self, ibo_val_db):
        self.my_array.update_distortion(ibo_db=ibo_val_db, avg_sample_pow=self.my_mod.avg_sample_power)
        if isinstance(self.my_cnc_rx, corrector.CncReceiver):
            self.my_cnc_rx.update_distortion(ibo_db=ibo_val_db)
        self.ibo_val_db = self.my_array.array_elements[0].impairment.ibo_db
        self.recalculate_agc(ak_part_only=True)

    def set_snr(self, snr_db_val):
        self.my_noise.snr_db = snr_db_val

    def set_precoding_and_recalculate_agc(self):
        self.my_array.set_precoding_matrix(channel_mat_fd=self.my_miso_chan.channel_mat_fd, mr_precoding=True)
        self.recalculate_agc()

    def recalculate_agc(self, ak_part_only=False):
        if not ak_part_only:
            hk_mat = np.concatenate((self.my_miso_chan.channel_mat_fd[:, -self.my_mod.n_sub_carr // 2:],
                                     self.my_miso_chan.channel_mat_fd[:, 1:(self.my_mod.n_sub_carr // 2) + 1]), axis=1)
            vk_mat = self.my_array.get_precoding_mat()
            self.vk_pow_vec = np.sum(np.power(np.abs(vk_mat), 2), axis=1)
            self.hk_vk_agc = np.multiply(hk_mat, vk_mat)
            hk_vk_agc_avg_vec = np.sum(self.hk_vk_agc, axis=0)
            self.hk_vk_noise_scaler = np.mean(np.power(np.abs(hk_vk_agc_avg_vec), 2))

            self.hk_vk_agc_nfft = np.ones(self.my_mod.n_fft, dtype=np.complex128)
            self.hk_vk_agc_nfft[-(self.n_sub_carr // 2):] = hk_vk_agc_avg_vec[0:self.n_sub_carr // 2]
            self.hk_vk_agc_nfft[1:(self.n_sub_carr // 2) + 1] = hk_vk_agc_avg_vec[self.n_sub_carr // 2:]

            self.my_array.update_distortion(ibo_db=self.ibo_val_db, avg_sample_pow=self.my_mod.avg_sample_power)
            if isinstance(self.my_cnc_rx, corrector.CncReceiver):
                self.my_cnc_rx.update_distortion(ibo_db=self.ibo_val_db)

        ibo_vec = 10 * np.log10(10 ** (self.ibo_val_db / 10) * self.my_mod.n_sub_carr /
                                (self.vk_pow_vec * self.n_ant_val))
        ak_vect = self.my_mod.calc_alpha(ibo_db=ibo_vec)
        ak_vect = np.expand_dims(ak_vect, axis=1)

        ak_hk_vk_agc = ak_vect * self.hk_vk_agc
        ak_hk_vk_agc_avg_vec = np.sum(ak_hk_vk_agc, axis=0)
        self.ak_hk_vk_noise_scaler = np.mean(np.power(np.abs(ak_hk_vk_agc_avg_vec), 2))

        self.ak_hk_vk_agc_nfft = np.ones(self.my_mod.n_fft, dtype=np.complex128)
        self.ak_hk_vk_agc_nfft[-(self.n_sub_carr // 2):] = ak_hk_vk_agc_avg_vec[0:self.n_sub_carr // 2]
        self.ak_hk_vk_agc_nfft[1:(self.n_sub_carr // 2) + 1] = ak_hk_vk_agc_avg_vec[self.n_sub_carr // 2:]

        if isinstance(self.my_cnc_rx, corrector.McncReceiver):
            self.my_cnc_rx.agc_corr_vec = self.ak_hk_vk_agc_nfft
