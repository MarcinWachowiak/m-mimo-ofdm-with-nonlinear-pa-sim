import numpy as np

import distortion
import utilities


class Transceiver:
    def __init__(self, modem, center_freq=None, carrier_spacing=None, impairment=None, cord_x=0, cord_y=0, cord_z=0):
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

    def set_position(self, cord_x, cord_y, cord_z):
        self.cord_x = cord_x
        self.cord_y = cord_y
        self.cord_z = cord_z

    def set_ant_gains(self, tx_ant_gain_db, rx_ant_gain_db):
        self.tx_ant_gain_db = tx_ant_gain_db
        self.rx_ant_gain_db = rx_ant_gain_db

    def set_tx_power_dbm(self, tx_power_dbm):
        self.tx_power_dbm = tx_power_dbm

    def correct_constellation(self):
        self.modem.correct_constellation(ibo_db=self.impairment.ibo_db)

    def update_distortion(self, ibo_db):
        self.impairment.set_ibo(ibo_db=ibo_db)
        self.modem.update_alpha(ibo_db=ibo_db)

    # legacy method -  in case amplification of signal is required (dBm scale)
    # def tx_amplify(self, in_sig):
    #     return in_sig * np.sqrt(1e-3 * 10 ** (self.tx_power_dbm / 10) / self.modem.avg_sample_power)

    def transmit(self, in_bits, out_domain_fd=True, skip_dist=False, return_both=False):
        clean_symb_td = self.modem.modulate(in_bits)
        if skip_dist or self.impairment is None:
            if out_domain_fd:
                return utilities.to_freq_domain(clean_symb_td, remove_cp=True, cp_len=self.modem.cp_len)
            else:
                return clean_symb_td
        elif return_both and self.impairment is not None:
            if out_domain_fd:
                return utilities.to_freq_domain(self.impairment.process(clean_symb_td), remove_cp=True,
                                                cp_len=self.modem.cp_len), \
                       utilities.to_freq_domain(clean_symb_td, remove_cp=True, cp_len=self.modem.cp_len)
            else:
                return self.impairment.process(clean_symb_td), clean_symb_td
        elif self.impairment is not None:
            if out_domain_fd:
                return utilities.to_freq_domain(self.impairment.process(clean_symb_td), remove_cp=True,
                                                cp_len=self.modem.cp_len)
            else:
                return self.impairment.process(clean_symb_td)

    def receive(self, in_symb):
        return self.modem.demodulate(in_symb)
