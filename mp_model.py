import copy

import numpy as np

import channel
import corrector
import utilities

from modulation import OfdmQamModem
from antenna_array import AntennaArray
from transceiver import Transceiver
from multiprocessing import Array
from numpy import ndarray


class Link():
    """
    Wireless link class used in parallel simulation of the wireless system.

    :param mod_obj: OFDM modem object used in the system
    :param array_obj: antenna array object used in the system
    :param std_rx_obj: transceiver serving as a receiver
    :param chan_obj: channel object used in the system
    :param noise_obj: noise object simulating the noise in the receiver
    :param rx_loc_var: variance of the receiver position - used to average the channel effects
    :param n_err_min: minimum number of bit errors in the BER simulation
    :param bits_sent_max: maximum number of sent bits in the BER simulation
    :param is_mcnc: flag if the receiver is the MCNC or CNC version
    :param csi_epsylon: value of the channel state information (CSI) error, if 'None' or 0 CSI error is not present
    """

    def __init__(self, mod_obj: OfdmQamModem, array_obj: AntennaArray, std_rx_obj: Transceiver, chan_obj, noise_obj,
                 rx_loc_var: float, n_err_min: int, bits_sent_max: int, is_mcnc: bool = False,
                 csi_epsylon: float = None):
        """
        Create a wireless link object.
        """
        self.my_mod = copy.deepcopy(mod_obj)
        self.my_array = copy.deepcopy(array_obj)
        self.my_standard_rx = copy.deepcopy(std_rx_obj)

        self.rx_loc_x = self.my_standard_rx.cord_x
        self.rx_loc_y = self.my_standard_rx.cord_y

        self.my_noise = copy.deepcopy(noise_obj)
        self.my_csi_noise = copy.deepcopy(noise_obj)
        self.csi_epsylon = csi_epsylon

        if isinstance(chan_obj, channel.MisoQuadrigaFd):
            self.is_quadriga = True
            self.channel_model_str = chan_obj.channel_model_str
            # some dummy channels needed for setup
            my_miso_los_chan = channel.MisoLosFd()
            my_miso_los_chan.calc_channel_mat(tx_transceivers=array_obj.array_elements, rx_transceiver=std_rx_obj,
                                              skip_attenuation=False)
            self.my_miso_chan = my_miso_los_chan
            if self.csi_epsylon is not None:
                self.my_miso_chan_csi_err = my_miso_los_chan
        else:
            self.is_quadriga = False
            self.my_miso_chan = copy.deepcopy(chan_obj)
            if self.csi_epsylon is not None:
                self.my_miso_chan_csi_err = copy.deepcopy(self.my_miso_chan)

        if is_mcnc:
            if self.csi_epsylon is None:
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

    def simulate(self, incl_clean_run: bool, reroll_chan: bool, cnc_n_iter_lst: list, seed_arr: list,
                 n_err_shared_arr: Array, n_bits_sent_shared_arr: Array) -> None:
        """
        Run the wireless link simulation.

        :param incl_clean_run: flag it to include the run with no distortion
        :param reroll_chan: flag if to relocate the receiver to average the channel effects
        :param cnc_n_iter_lst: list of the number of CNC/MCNC iterations to perform
        :param seed_arr: list of the seeds for the random number generators in the simulation
        :param n_err_shared_arr: array containing the number of bit errors per iteration - shared across link objects
        :param n_bits_sent_shared_arr: array containing the number of bits sent per iteration - shared across
            link objects
        :return: None
        """

        # matlab engine is not serializable and has to be started inside the process function
        if self.is_quadriga:
            self.my_miso_chan = channel.MisoQuadrigaFd(tx_transceivers=self.my_array.array_elements,
                                                       rx_transceiver=self.my_standard_rx,
                                                       channel_model_str=self.channel_model_str)
            if self.csi_epsylon is not None:
                self.my_miso_chan_csi_err = channel.MisoQuadrigaFd(tx_transceivers=self.my_array.array_elements,
                                                                   rx_transceiver=self.my_standard_rx,
                                                                   channel_model_str=self.channel_model_str,
                                                                   start_matlab_eng=False)
        # update MCNC channel
        if isinstance(self.my_cnc_rx, corrector.McncReceiver):
            if self.csi_epsylon is None:
                self.my_cnc_rx.channel = self.my_miso_chan
            else:
                self.my_cnc_rx.channel = self.my_miso_chan_csi_err

        self.bit_rng = np.random.default_rng(seed_arr[0])
        self.my_noise.rng_gen = np.random.default_rng(seed_arr[1])
        self.loc_rng = np.random.default_rng(seed_arr[2])
        if self.csi_epsylon is not None:
            self.my_csi_noise.rng_gen = np.random.default_rng(seed_arr[3])

        # if isinstance(self.my_miso_chan, channel.MisoQuadrigaFd):
        #     self.my_miso_chan.meng.rng(seed_arr[4].astype(np.uint32))
        #     if self.csi_epsylon is not None:
        #         self.my_miso_chan_csi_err.meng.rng(seed_arr[4].astype(np.uint32))

        res_idx = 0
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

        # # close the matlab engine process
        # if self.is_quadriga:
        #     self.my_miso_chan.meng.quit()
        #     if self.csi_epsylon is not None:
        #         self.my_miso_chan_csi_err.meng.quit()

    def update_distortion(self, ibo_val_db: float) -> None:
        """
        Update the parameters of distortion in the link object, update the alpha and equalization vector.

        :param ibo_val_db: input back-off value in [dB]
        :return: None
        """
        self.my_array.update_distortion(ibo_db=ibo_val_db, avg_sample_pow=self.my_mod.avg_sample_power)
        if isinstance(self.my_cnc_rx, corrector.CncReceiver):
            self.my_cnc_rx.update_distortion(ibo_db=ibo_val_db)
        self.ibo_val_db = self.my_array.array_elements[0].impairment.ibo_db
        self.recalculate_agc(ak_part_only=True)

    def set_snr(self, snr_db_val: float) -> None:
        """
        Set the signal-to-noise ratio of the Noise object in the link.

        :param snr_db_val: signal-to-noise ratio expressed in [dB]
        :return: None
        """

        self.my_noise.snr_db = snr_db_val

    def set_precoding_and_recalculate_agc(self) -> None:
        """
        Set the precoding matrix based on the channel object and update the equalization vector inside the link.

        :return: None
        """

        if self.csi_epsylon is None:
            self.my_array.set_precoding_matrix(channel_mat_fd=self.my_miso_chan.channel_mat_fd, mr_precoding=True)
            self.recalculate_agc(channel_mat_fd=self.my_miso_chan.channel_mat_fd)
        else:
            # introduce csi errors in the channel model used in precoding and MCNC algorithm
            noisy_channel_mat_fd = np.copy(self.my_miso_chan.channel_mat_fd)

            # antenna wise noise addition
            for row_idx, chan_per_ant in enumerate(noisy_channel_mat_fd):
                sc_chan_per_ant = np.concatenate(
                    (chan_per_ant[-self.my_mod.n_sub_carr // 2:], chan_per_ant[1:(self.my_mod.n_sub_carr // 2) + 1]))
                channel_power_per_fd_sample = np.sum(np.abs(sc_chan_per_ant) ** 2) / len(sc_chan_per_ant)
                channel_noise = self.my_noise.rng_gen.standard_normal((len(sc_chan_per_ant), 2)).view(np.complex128)[:,
                                0] * 0.5 * np.sqrt(2 * channel_power_per_fd_sample) * self.csi_epsylon
                scaled_channel = np.sqrt(1 - np.power(self.csi_epsylon, 2)) * sc_chan_per_ant
                noisy_sc_chan_row = scaled_channel + channel_noise
                # noisy_channel_power_per_fd_sample = np.sum(np.abs(noisy_sc_chan_row) ** 2) / len(sc_chan_per_ant)

                # noisy_sc_chan_row = self.my_csi_noise.process(in_sig=sc_chan_per_ant, avg_sample_pow=channel_power_per_fd_sample, disp_data=False)
                noisy_channel_mat_fd[row_idx, -(self.my_mod.n_sub_carr // 2):] = noisy_sc_chan_row[
                                                                                 0:self.my_mod.n_sub_carr // 2]
                noisy_channel_mat_fd[row_idx, 1:(self.my_mod.n_sub_carr // 2) + 1] = noisy_sc_chan_row[
                                                                                     self.my_mod.n_sub_carr // 2:]

            self.my_miso_chan_csi_err.channel_mat_fd = noisy_channel_mat_fd

            self.my_array.set_precoding_matrix(channel_mat_fd=self.my_miso_chan_csi_err.channel_mat_fd,
                                               mr_precoding=True)
            self.recalculate_agc(channel_mat_fd=self.my_miso_chan_csi_err.channel_mat_fd)

    def recalculate_agc(self, channel_mat_fd: ndarray = None, ak_part_only: bool = False) -> None:
        """
        Recalculate the equalization vector once the precoding has been changed.

        :param channel_mat_fd: matrix containing channel coefficients
        :param ak_part_only: flag if only the alpha shrinking coefficient was changed
        :return: None
        """
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
