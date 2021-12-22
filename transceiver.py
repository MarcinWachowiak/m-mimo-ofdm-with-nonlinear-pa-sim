import utilities


class Transceiver:
    def __init__(self, modem, center_freq=None, carrier_spacing=None, impairment=None, cord_x=0, cord_y=0, cord_z=0):
        self.modem = modem
        self.impairment = impairment
        self.center_freq = center_freq
        self.carrier_spacing = carrier_spacing
        # antenna gains
        self.tx_ant_gain_db = 0
        self.rx_ant_gain_db = 0
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

    def transmit(self, in_bits, out_domain_fd=True, skip_dist=False, return_both=False):
        clean_symb_td = self.modem.modulate(in_bits)

        if skip_dist or self.impairment is None:
            if out_domain_fd:
                return utilities.to_freq_domain(clean_symb_td, remove_cp=True, cp_len=self.modem.cp_len)
            else:
                return clean_symb_td
        elif return_both and self.impairment is not None:
            if out_domain_fd:
                return utilities.to_freq_domain(self.impairment.process(clean_symb_td), remove_cp=True, cp_len=self.modem.cp_len),\
                       utilities.to_freq_domain(clean_symb_td, remove_cp=True, cp_len=self.modem.cp_len)
            else:
                return self.impairment.process(clean_symb_td), clean_symb_td
        elif self.impairment is not None:
            if out_domain_fd:
                utilities.to_freq_domain(self.impairment.process(clean_symb_td), remove_cp=True,
                                         cp_len=self.modem.cp_len)
            else:
                return self.impairment.process(clean_symb_td)

    def receive(self, in_symb, skip_noise=False):
        # add noise to the received signal accordingly to RX noise level
        return self.modem.demodulate(in_symb)
