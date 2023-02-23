import numpy as np

import distortion
import utilities

from modulation import OfdmQamModem
from numpy import ndarray

class Transceiver:
    """
        Radio transceiver class, includes modem and nonlinear power-amplifier model.

        :param modem: OFDM modem object
        :param center_freq: center frequency
        :param carrier_spacing: carrier spacing
        :param impairment: distortion object
        :param cord_x: X coordinates of the transceiver [m]
        :param cord_y: Y coordinates of the transceiver [m]
        :param cord_z: Z coordinates of the transceiver [m]
    """
    def __init__(self, modem: OfdmQamModem, center_freq: int, carrier_spacing: int, impairment, cord_x: float = 0,
                 cord_y: float = 0, cord_z: float = 0):
        """
        Create a transceiver object.
        """

        self.modem = modem
        self.impairment = impairment
        self.center_freq = center_freq
        self.carrier_spacing = carrier_spacing
        # update modem alpha in regard to impairment object
        if isinstance(impairment, distortion.SoftLimiter):
            self.modem.update_alpha(ibo_db=impairment.ibo_db)
        # antenna gains
        self.tx_ant_gain_db = 0
        self.rx_ant_gain_db = 0
        # TX power
        # default value for legacy simulations
        self.tx_power_dbm = 10 * np.log10(1000 * self.modem.avg_sample_power)
        # position of transceiver/antenna
        self.cord_x = cord_x
        self.cord_y = cord_y
        self.cord_z = cord_z

    def set_position(self, cord_x: float, cord_y: float, cord_z: float) -> None:
        """
        Set the physical position of the transceiver object.

        :param cord_x: X coordinates of the transceiver [m]
        :param cord_y: Y coordinates of the transceiver [m]
        :param cord_z: Z coordinates of the transceiver [m]
        :return: None
        """
        self.cord_x = cord_x
        self.cord_y = cord_y
        self.cord_z = cord_z

    def set_ant_gains(self, tx_ant_gain_db: float, rx_ant_gain_db: float) -> None:
        """
        Set the antenna gains.

        :param tx_ant_gain_db: transmit antenna gain in [dB]
        :param rx_ant_gain_db: receive antenna gain in [dB]
        :return: None
        """
        self.tx_ant_gain_db = tx_ant_gain_db
        self.rx_ant_gain_db = rx_ant_gain_db

    def set_tx_power_dbm(self, tx_power_dbm: float) -> None:
        """
        Set the transmit power in [dBm]
        :param tx_power_dbm: transmit power in [dBm]
        :return: None
        """
        self.tx_power_dbm = tx_power_dbm

    def correct_constellation(self) -> None:
        """
        Correct the reference constellation with the alpha shrinking coefficient according to the distortion parameters.
        :return: None
        """
        self.modem.correct_constellation(ibo_db=self.impairment.ibo_db)

    def update_distortion(self, ibo_db: float) -> None:
        """
        Update the parameters of distortion in the transceiver object.

        :param ibo_db: input back-off value expressed in [dB]
        :return: None
        """
        self.impairment.set_ibo(ibo_db=ibo_db)
        self.modem.update_alpha(ibo_db=ibo_db)

    # legacy method -  in case amplification of signal is required (dBm scale)
    # def tx_amplify(self, in_sig):
    #     return in_sig * np.sqrt(1e-3 * 10 ** (self.tx_power_dbm / 10) / self.modem.avg_sample_power)

    def transmit(self, in_bits: ndarray, out_domain_fd: bool = True, skip_dist: bool = False, return_both: bool = False,
                 sum_usr_signals: bool = True):
        """
        Transmit the input bits: modulate, precode and process by the nonlinearity model.

        :param in_bits: input data bits vector
        :param out_domain_fd: flag if the output should be in frequency domain or in time domain
        :param skip_dist: flag if to skip the processing by the nonlinear distortion model
        :param return_both: flag if both the distorted and nondistorted signals should be output
        :param sum_usr_signals: flag if the users signals should be summed in multi-user scenario
        :return: vector of the transmitted signal
        """

        clean_symb_td = self.modem.modulate(in_bits, sum_usr_signals=sum_usr_signals)
        if skip_dist or self.impairment is None:
            if out_domain_fd:
                if sum_usr_signals:
                    return utilities.to_freq_domain(clean_symb_td, remove_cp=True, cp_len=self.modem.cp_len)
                else:
                    per_usr_sig = []
                    for usr_idx in range(self.modem.n_users):
                        per_usr_sig.append(
                            utilities.to_freq_domain(clean_symb_td[usr_idx], remove_cp=True, cp_len=self.modem.cp_len))
                    return per_usr_sig
            else:
                if sum_usr_signals:
                    return clean_symb_td
                else:
                    per_usr_sig = []
                    for usr_idx in range(self.modem.n_users):
                        per_usr_sig.append(clean_symb_td[usr_idx])
                    return per_usr_sig


        elif return_both and self.impairment is not None:
            if out_domain_fd:
                if sum_usr_signals:
                    return utilities.to_freq_domain(self.impairment.process(clean_symb_td), remove_cp=True,
                                                    cp_len=self.modem.cp_len), utilities.to_freq_domain(clean_symb_td,
                                                                                                        remove_cp=True,
                                                                                                        cp_len=self.modem.cp_len)
                else:
                    per_usr_sig = []
                    for usr_idx in range(self.modem.n_users):
                        per_usr_sig.append([utilities.to_freq_domain(self.impairment.process(clean_symb_td[usr_idx]),
                                                                     remove_cp=True, cp_len=self.modem.cp_len),
                                            utilities.to_freq_domain(clean_symb_td[usr_idx], remove_cp=True,
                                                                     cp_len=self.modem.cp_len)])
                    return per_usr_sig
            else:
                if sum_usr_signals:
                    return self.impairment.process(clean_symb_td), clean_symb_td
                else:
                    per_usr_sig = []
                    for usr_idx in range(self.modem.n_users):
                        per_usr_sig.append([self.impairment.process(clean_symb_td[usr_idx]), clean_symb_td[usr_idx]])
                    return per_usr_sig
        elif self.impairment is not None:
            if out_domain_fd:
                if sum_usr_signals:
                    return utilities.to_freq_domain(self.impairment.process(clean_symb_td), remove_cp=True,
                                                    cp_len=self.modem.cp_len)
                else:
                    per_usr_sig = []
                    for usr_idx in range(self.modem.n_users):
                        per_usr_sig.append(
                            utilities.to_freq_domain(self.impairment.process(clean_symb_td[usr_idx]), remove_cp=True,
                                                     cp_len=self.modem.cp_len))
                    return per_usr_sig
            else:
                if sum_usr_signals:
                    return self.impairment.process(clean_symb_td)
                else:
                    per_usr_sig = []
                    for usr_idx in range(self.modem.n_users):
                        per_usr_sig.append(self.impairment.process(clean_symb_td[usr_idx]))
                    return per_usr_sig

    def receive(self, in_symb: ndarray):
        """
        Receive the signal: demodulate and decode.

        :param in_symb: input signal vector
        :return: received data bits array
        """

        return self.modem.demodulate(in_symb)
