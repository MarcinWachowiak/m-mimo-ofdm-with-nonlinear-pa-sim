"""
Measure the signal-to-distortion ratio as a function of input back-off for selected number of antennas and channels.
"""
import os
import sys
from multiprocessing import Array, Value

sys.path.append(os.getcwd())

from antenna_array import AntennaArray
from modulation import OfdmQamModem
from transceiver import Transceiver

import copy

import numpy as np

import channel


class Sdr_vs_ibo_vs_chan_link():
    """
    Wireless link class used in parallel simulation of the wireless system.

    :param mod_obj: OFDM modem object used in the system
    :param array_obj: antenna array object used in the system
    :param std_rx_obj: transceiver serving as a receiver
    :param chan_obj: channel object used in the system
    :param rx_loc_var: variance of the receiver position - used to average the channel effects
    """

    def __init__(self, mod_obj: OfdmQamModem, array_obj: AntennaArray, std_rx_obj: Transceiver, chan_obj,
                 rx_loc_var: float):
        """
        Create a wireless link object.
        """
        self.my_mod = copy.deepcopy(mod_obj)
        self.my_array = copy.deepcopy(array_obj)
        self.my_standard_rx = copy.deepcopy(std_rx_obj)

        self.rx_loc_x = self.my_standard_rx.cord_x
        self.rx_loc_y = self.my_standard_rx.cord_y

        if isinstance(chan_obj, channel.MisoQuadrigaFd):
            self.is_quadriga = True
            self.channel_model_str = chan_obj.channel_model_str
            # some dummy channels needed for setup
            my_miso_los_chan = channel.MisoLosFd()
            my_miso_los_chan.calc_channel_mat(tx_transceivers=array_obj.array_elements, rx_transceiver=std_rx_obj,
                                              skip_attenuation=False)
            self.my_miso_chan = my_miso_los_chan

        else:
            self.is_quadriga = False
            self.my_miso_chan = copy.deepcopy(chan_obj)

        self.loc_rng = np.random.default_rng(1)
        self.bit_rng = np.random.default_rng(2)
        self.rx_loc_var = rx_loc_var

        self.n_ant_val = len(self.my_array.array_elements)
        self.n_bits_per_ofdm_sym = self.my_mod.n_bits_per_ofdm_sym
        self.n_sub_carr = self.my_mod.n_sub_carr
        self.ibo_val_db = self.my_array.array_elements[0].impairment.ibo_db

        self.my_array.set_precoding_matrix(channel_mat_fd=self.my_miso_chan.channel_mat_fd,
                                           mr_precoding=True)
        self.my_array.update_distortion(ibo_db=self.ibo_val_db, avg_sample_pow=self.my_mod.avg_sample_power)

    def simulate(self, reroll_chan: bool, seed_arr: list, n_snap_idx: Value,
                 sdr_at_ibo_per_symb_arr: Array) -> None:
        """
        Run the wireless link simulation.

        :param n_snap_idx:
        :param sdr_at_ibo_per_symb_arr:
        :param reroll_chan: flag if to relocate the receiver to average the channel effects
        :param seed_arr: list of the seeds for the random number generators in the simulation

        :return: None
        """
        # matlab engine is not serializable and has to be started inside the process function
        if self.is_quadriga:
            self.my_miso_chan = channel.MisoQuadrigaFd(tx_transceivers=self.my_array.array_elements,
                                                       rx_transceiver=self.my_standard_rx,
                                                       channel_model_str=self.channel_model_str)

        self.bit_rng = np.random.default_rng(seed_arr[0])
        self.loc_rng = np.random.default_rng(seed_arr[1])

        # if isinstance(self.my_miso_chan, channel.MisoQuadrigaFd):
        #     self.my_miso_chan.meng.rng(seed_arr[4].astype(np.uint32))
        #     if self.csi_epsylon is not None:
        #         self.my_miso_chan_csi_err.meng.rng(seed_arr[4].astype(np.uint32))

        n_snap_idx_val = (n_snap_idx.get_obj()).value
        sdr_at_ibo_per_symb_arr_np = np.frombuffer(sdr_at_ibo_per_symb_arr.get_obj())

        while True:
            if not int(n_snap_idx_val) < len(sdr_at_ibo_per_symb_arr_np):
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

                self.my_array.set_precoding_matrix(channel_mat_fd=self.my_miso_chan.channel_mat_fd,
                                                   mr_precoding=True)
                self.my_array.update_distortion(ibo_db=self.ibo_val_db, avg_sample_pow=self.my_mod.avg_sample_power)

            vk_mat = self.my_array.get_precoding_mat()
            vk_pow_vec = np.sum(np.power(np.abs(vk_mat), 2), axis=1)

            ibo_vec = 10 * np.log10(10 ** (self.ibo_val_db / 10) * self.my_mod.n_sub_carr /
                                    (vk_pow_vec * self.n_ant_val))
            ak_vect = self.my_mod.calc_alpha(ibo_db=ibo_vec)
            ak_vect = np.expand_dims(ak_vect, axis=1)

            tx_bits = self.bit_rng.choice((0, 1), self.n_bits_per_ofdm_sym)
            arr_tx_sig_fd, clean_sig_mat_fd = self.my_array.transmit(tx_bits, out_domain_fd=True, skip_dist=False,
                                                                     return_both=True)

            rx_sig_fd = self.my_miso_chan.propagate(in_sig_mat=arr_tx_sig_fd, sum_signals=False)
            rx_sc_ofdm_symb_fd = np.concatenate(
                (rx_sig_fd[:, -self.my_mod.n_sub_carr // 2:], rx_sig_fd[:, 1:(self.my_mod.n_sub_carr // 2) + 1]),
                axis=1)
            # rx_sc_ofdm_symb_td = utilities.to_time_domain(rx_sc_ofdm_symb_fd)

            clean_rx_sig_fd = self.my_miso_chan.propagate(in_sig_mat=clean_sig_mat_fd, sum_signals=False)
            clean_sc_ofdm_symb_fd = np.concatenate((clean_rx_sig_fd[:, -self.my_mod.n_sub_carr // 2:],
                                                    clean_rx_sig_fd[:, 1:(self.my_mod.n_sub_carr // 2) + 1]), axis=1)

            sc_ofdm_distortion_sig = np.subtract(rx_sc_ofdm_symb_fd, (ak_vect * clean_sc_ofdm_symb_fd))

            desired_sig_pow = np.sum(np.power(np.abs(np.sum(ak_vect * clean_sc_ofdm_symb_fd, axis=0)), 2))
            distortion_sig_pow = np.sum(np.power(np.abs(np.sum(sc_ofdm_distortion_sig, axis=0)), 2))
            # calculate SDR on symbol basis
            sdr_at_ibo_per_symb_arr_np[n_snap_idx_val] = (desired_sig_pow / distortion_sig_pow)
            n_snap_idx_val += 1

    def update_distortion(self, ibo_val_db):
        self.ibo_val_db = ibo_val_db
        self.my_array.update_distortion(ibo_db=ibo_val_db, avg_sample_pow=self.my_mod.avg_sample_power)
