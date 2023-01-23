import copy

import numpy as np

import channel
import corrector
import utilities
import matlab.engine


class Link_LDPC():
    def __init__(self, mod_obj, array_obj, std_rx_obj, chan_obj, noise_obj, rx_loc_var, n_err_min,
                 bits_sent_max, code_rate, max_ldpc_ite, is_mcnc=False, csi_noise_db=None, ):
        self.my_mod = copy.deepcopy(mod_obj)
        self.my_array = copy.deepcopy(array_obj)
        self.my_standard_rx = copy.deepcopy(std_rx_obj)

        self.rx_loc_x = self.my_standard_rx.cord_x
        self.rx_loc_y = self.my_standard_rx.cord_y

        self.my_noise = copy.deepcopy(noise_obj)
        self.my_csi_noise = copy.deepcopy(noise_obj)
        self.csi_noise_db = csi_noise_db
        self.my_csi_noise.snr_db = self.csi_noise_db
        self.code_rate = code_rate
        self.max_ldpc_ite = max_ldpc_ite


        if isinstance(chan_obj, channel.MisoQuadrigaFd):
            self.is_quadriga = True
            self.channel_model_str = chan_obj.channel_model_str
            # some dummy channels needed for setup
            my_miso_los_chan = channel.MisoLosFd()
            my_miso_los_chan.calc_channel_mat(tx_transceivers=array_obj.array_elements, rx_transceiver=std_rx_obj,
                                              skip_attenuation=False)
            self.my_miso_chan = my_miso_los_chan
            if self.csi_noise_db is not None:
                self.my_miso_chan_csi_err = my_miso_los_chan
        else:
            self.is_quadriga = False
            self.my_miso_chan = copy.deepcopy(chan_obj)
            if self.csi_noise_db is not None:
                self.my_miso_chan_csi_err = copy.deepcopy(self.my_miso_chan)

        if is_mcnc:
            if self.csi_noise_db is None:
                self.my_cnc_rx = corrector.McncReceiver(self.my_array, self.my_miso_chan)
            else:
                self.my_cnc_rx = corrector.McncReceiver(self.my_array, self.my_miso_chan_csi_err)
        else:
            self.my_cnc_rx = corrector.CncReceiver(copy.deepcopy(array_obj.base_transceiver.modem),
                                                   copy.deepcopy(array_obj.base_transceiver.impairment))

        self.my_noise.rng_gen = np.random.default_rng(0)
        self.loc_rng = np.random.default_rng(1)
        self.bit_rng = np.random.default_rng(2)
        self.my_csi_noise.rng_gen = np.random.default_rng(3)
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

        # matlab engine is not serializable and has to be started inside the process function
        if self.is_quadriga:
            self.my_miso_chan = channel.MisoQuadrigaFd(tx_transceivers=self.my_array.array_elements, rx_transceiver=self.my_standard_rx, channel_model_str=self.channel_model_str)
            if self.csi_noise_db is not None:
                self.my_miso_chan_csi_err = channel.MisoQuadrigaFd(tx_transceivers=self.my_array.array_elements, rx_transceiver=self.my_standard_rx, channel_model_str=self.channel_model_str)
        # update MCNC channel
        if isinstance(self.my_cnc_rx, corrector.McncReceiver):
            if self.csi_noise_db is None:
                self.my_cnc_rx.channel = self.my_miso_chan
            else:
                self.my_cnc_rx.channel = self.my_miso_chan_csi_err

        meng = matlab.engine.start_matlab()
        meng.rng(7312)

        self.rv = meng.double(0)
        self.modulation_format_str = "64QAM"
        self.bits_per_symbol = 6
        self.n_layers = meng.double(1)

        self.n_info_bits = meng.int64(self.my_mod.n_bits_per_ofdm_sym * self.code_rate)
        self.out_len = meng.double(np.ceil(self.n_info_bits / self.code_rate))
        if self.out_len != self.my_mod.n_bits_per_ofdm_sym:
            raise ValueError('Code output length does not match modulator input length!')

        self.cbs_info_dict = meng.nrDLSCHInfo(self.n_info_bits, self.code_rate)

        self.bit_rng = np.random.default_rng(seed_arr[0])
        self.my_noise.rng_gen = np.random.default_rng(seed_arr[1])
        self.loc_rng = np.random.default_rng(seed_arr[2])
        if self.csi_noise_db is not None:
            self.my_csi_noise.rng_gen = np.random.default_rng(seed_arr[3])

        if isinstance(self.my_miso_chan, channel.MisoQuadrigaFd):
            self.my_miso_chan.meng.rng(seed_arr[4].astype(np.uint32))
            if self.csi_noise_db is not None:
                self.my_miso_chan_csi_err.meng.rng(seed_arr[4].astype(np.uint32))

        if self.code_rate is not None:
            meng.rng(seed_arr[5].astype(np.uint32))

        res_idx = 0
        noise_var_snr_based = np.complex128(2 * self.my_mod.avg_symbol_power / (10 ** (self.my_noise.snr_db / 10)))
        if incl_clean_run:
            res_idx = 1
            # clean RX run
            while True:
                if np.logical_and((n_err_shared_arr[0] < self.n_err_min),
                                  (n_bits_sent_shared_arr[0] < self.bits_sent_max)):

                    if reroll_chan:
                        if not isinstance(self.my_miso_chan, channel.MisoRayleighFd):
                            # reroll location
                            self.my_standard_rx.set_position(
                                cord_x=self.rx_loc_x + self.loc_rng.uniform(low=-self.rx_loc_var / 2.0,
                                                                            high=self.rx_loc_var / 2.0),
                                cord_y=self.rx_loc_x + self.loc_rng.uniform(low=-self.rx_loc_var / 2.0,
                                                                            high=self.rx_loc_var / 2.0),
                                cord_z=self.my_standard_rx.cord_z)
                            self.my_miso_chan.calc_channel_mat(tx_transceivers=self.my_array.array_elements,
                                                               rx_transceiver=self.my_standard_rx)
                        # elif isinstance(self.my_miso_chan, channel.MisoRandomPathsFd):
                        #     self.my_miso_chan.reroll_channel_coeffs(tx_transceivers=self.my_array.array_elements)
                        else:
                            self.my_miso_chan.reroll_channel_coeffs()

                        self.set_precoding_and_recalculate_agc()

                    tx_bits = self.bit_rng.choice((0, 1), self.n_info_bits)

                    crc_encoded_bits = meng.nrCRCEncode(meng.int8(meng.transpose(tx_bits)), self.cbs_info_dict['CRC'])
                    code_block_segment_in = meng.nrCodeBlockSegmentLDPC(crc_encoded_bits, self.cbs_info_dict['BGN'])
                    ldpc_encoded_bits = meng.nrLDPCEncode(code_block_segment_in, self.cbs_info_dict['BGN'])
                    rm_ldpc_bits = np.squeeze(np.array(meng.nrRateMatchLDPC(ldpc_encoded_bits, self.out_len, self.rv, self.modulation_format_str, self.n_layers)))

                    clean_ofdm_symbol = self.my_array.transmit(rm_ldpc_bits, out_domain_fd=True, return_both=False,
                                                               skip_dist=True)
                    clean_rx_ofdm_symbol = self.my_miso_chan.propagate(in_sig_mat=clean_ofdm_symbol)
                    clean_rx_ofdm_symbol = self.my_noise.process(clean_rx_ofdm_symbol,
                                                                 avg_sample_pow=self.my_mod.avg_symbol_power * self.hk_vk_noise_scaler,
                                                                 disp_data=False)
                    clean_rx_ofdm_symbol = np.divide(clean_rx_ofdm_symbol, self.hk_vk_agc_nfft)
                    clean_rx_ofdm_symbol = utilities.to_time_domain(clean_rx_ofdm_symbol)
                    clean_rx_ofdm_symbol = np.concatenate((clean_rx_ofdm_symbol[-self.my_mod.cp_len:], clean_rx_ofdm_symbol))

                    rx_symbols = self.my_standard_rx.modem.demodulate(clean_rx_ofdm_symbol, get_symbols_only=True)
                    rx_llr_soft_bits = -self.my_standard_rx.modem.soft_detection_llr(rx_symbols, noise_var=noise_var_snr_based)
                    rate_recovered_bits = meng.nrRateRecoverLDPC(meng.transpose(meng.double(rx_llr_soft_bits)),
                                                                 self.n_info_bits, self.code_rate, self.rv, self.modulation_format_str,
                                                                 self.n_layers)
                    ldpc_decoded_bits = meng.nrLDPCDecode(rate_recovered_bits, self.cbs_info_dict['BGN'], self.max_ldpc_ite)
                    desegmented_bits = meng.nrCodeBlockDesegmentLDPC(ldpc_decoded_bits, self.cbs_info_dict['BGN'],
                                                                     self.n_info_bits + self.cbs_info_dict['L'])
                    rx_bits = np.squeeze(np.array(meng.transpose(meng.nrCRCDecode(desegmented_bits, self.cbs_info_dict['CRC']))))

                    n_bit_err = utilities.count_mismatched_bits(tx_bits, rx_bits)
                    n_err_shared_arr[0] += n_bit_err
                    n_bits_sent_shared_arr[0] += self.n_info_bits
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
                if not isinstance(self.my_miso_chan, channel.MisoRayleighFd):
                    # reroll location
                    self.my_standard_rx.set_position(
                        cord_x=self.rx_loc_x + self.loc_rng.uniform(low=-self.rx_loc_var / 2.0,
                                                                    high=self.rx_loc_var / 2.0),
                        cord_y=self.rx_loc_x + self.loc_rng.uniform(low=-self.rx_loc_var / 2.0,
                                                                    high=self.rx_loc_var / 2.0),
                        cord_z=self.my_standard_rx.cord_z)
                    self.my_miso_chan.calc_channel_mat(tx_transceivers=self.my_array.array_elements,
                                                       rx_transceiver=self.my_standard_rx)
                # elif isinstance(self.my_miso_chan, channel.MisoRandomPathsFd):
                #     self.my_miso_chan.reroll_channel_coeffs(tx_transceivers=self.my_array.array_elements)
                else:
                    self.my_miso_chan.reroll_channel_coeffs()

                self.set_precoding_and_recalculate_agc()

            tx_bits = self.bit_rng.choice((0, 1), self.n_info_bits)
            crc_encoded_bits = meng.nrCRCEncode(meng.int8(meng.transpose(tx_bits)), self.cbs_info_dict['CRC'])
            code_block_segment_in = meng.nrCodeBlockSegmentLDPC(crc_encoded_bits, self.cbs_info_dict['BGN'])
            ldpc_encoded_bits = meng.nrLDPCEncode(code_block_segment_in, self.cbs_info_dict['BGN'])
            rm_ldpc_bits = np.squeeze(np.array(
                meng.nrRateMatchLDPC(ldpc_encoded_bits, self.out_len, self.rv, self.modulation_format_str,
                                     self.n_layers)))

            tx_ofdm_symbol = self.my_array.transmit(rm_ldpc_bits, out_domain_fd=True, skip_dist=False)
            rx_ofdm_symbol = self.my_miso_chan.propagate(in_sig_mat=tx_ofdm_symbol)
            rx_ofdm_symbol = self.my_noise.process(rx_ofdm_symbol,
                                                   avg_sample_pow=self.my_mod.avg_symbol_power * self.ak_hk_vk_noise_scaler)

            rx_ofdm_symbol = np.divide(rx_ofdm_symbol, self.ak_hk_vk_agc_nfft)
            rx_sybmols_per_iter_lst = self.my_cnc_rx.receive(n_iters_lst=curr_ite_lst, in_sig_fd=rx_ofdm_symbol, return_bits=False)

            ber_idx = np.array(list(range(len(cnc_n_iter_lst))))
            act_ber_idx = ber_idx[ite_use_flags] + res_idx
            for idx in range(len(rx_sybmols_per_iter_lst)):
                rx_llr_soft_bits = -self.my_standard_rx.modem.soft_detection_llr(rx_sybmols_per_iter_lst[idx],
                                                                                 noise_var=noise_var_snr_based)
                rate_recovered_bits = meng.nrRateRecoverLDPC(meng.transpose(meng.double(rx_llr_soft_bits)),
                                                             self.n_info_bits, self.code_rate, self.rv,
                                                             self.modulation_format_str,
                                                             self.n_layers)
                ldpc_decoded_bits = meng.nrLDPCDecode(rate_recovered_bits, self.cbs_info_dict['BGN'], self.max_ldpc_ite)
                desegmented_bits = meng.nrCodeBlockDesegmentLDPC(ldpc_decoded_bits, self.cbs_info_dict['BGN'],
                                                                 self.n_info_bits + self.cbs_info_dict['L'])
                rx_bits = np.squeeze(
                    np.array(meng.transpose(meng.nrCRCDecode(desegmented_bits, self.cbs_info_dict['CRC']))))

                n_bit_err = utilities.count_mismatched_bits(tx_bits, rx_bits)
                n_err_shared_arr[act_ber_idx[idx]] += n_bit_err
                n_bits_sent_shared_arr[act_ber_idx[idx]] += self.n_info_bits

        # # close the matlab engine process
        # if self.is_quadriga:
        #     self.my_miso_chan.meng.quit()
        #     if self.csi_noise_db is not None:
        #         self.my_miso_chan_csi_err.meng.quit()

    def update_distortion(self, ibo_val_db):
        self.my_array.update_distortion(ibo_db=ibo_val_db, avg_sample_pow=self.my_mod.avg_sample_power)
        if isinstance(self.my_cnc_rx, corrector.CncReceiver):
            self.my_cnc_rx.update_distortion(ibo_db=ibo_val_db)
        self.ibo_val_db = self.my_array.array_elements[0].impairment.ibo_db
        self.recalculate_agc(ak_part_only=True)

    def set_snr(self, snr_db_val):
        self.my_noise.snr_db = snr_db_val

    def set_precoding_and_recalculate_agc(self):
        if self.csi_noise_db is None:
            self.my_array.set_precoding_matrix(channel_mat_fd=self.my_miso_chan.channel_mat_fd, mr_precoding=True)
            self.recalculate_agc(channel_mat_fd=self.my_miso_chan.channel_mat_fd)
        else:
            # introduce csi errors in the channel model used in precoding and MCNC algorithm
            noisy_channel_mat_fd = np.copy(self.my_miso_chan.channel_mat_fd)

            # antenna wise noise addition
            for row_idx, chan_per_ant in enumerate(noisy_channel_mat_fd):
                sc_chan_per_ant = np.concatenate((chan_per_ant[-self.my_mod.n_sub_carr // 2:],
                                     chan_per_ant[1:(self.my_mod.n_sub_carr // 2) + 1]))
                channel_power_per_fd_sample = np.sum(np.abs(sc_chan_per_ant) ** 2) / len(sc_chan_per_ant)
                noisy_sc_chan_row = self.my_csi_noise.process(in_sig=sc_chan_per_ant, avg_sample_pow=channel_power_per_fd_sample, disp_data=False)
                noisy_channel_mat_fd[row_idx, -(self.my_mod.n_sub_carr // 2):] = noisy_sc_chan_row[0:self.my_mod.n_sub_carr // 2]
                noisy_channel_mat_fd[row_idx, 1:(self.my_mod.n_sub_carr // 2) + 1] = noisy_sc_chan_row[self.my_mod.n_sub_carr // 2:]

            self.my_miso_chan_csi_err.channel_mat_fd = noisy_channel_mat_fd

            self.my_array.set_precoding_matrix(channel_mat_fd=self.my_miso_chan_csi_err.channel_mat_fd, mr_precoding=True)
            self.recalculate_agc(channel_mat_fd=self.my_miso_chan_csi_err.channel_mat_fd)

    def recalculate_agc(self, channel_mat_fd=None, ak_part_only=False):
        if not ak_part_only:
            hk_mat = np.concatenate((channel_mat_fd[:, -self.my_mod.n_sub_carr // 2:],
                                     channel_mat_fd[:, 1:(self.my_mod.n_sub_carr // 2) + 1]), axis=1)
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
