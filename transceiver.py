
class Transceiver:
    def __init__(self, modem, center_freq=None, carrier_spacing=None, impairment=None, cord_x=0, cord_y=0, cord_z=0):
        self.modem = modem
        self.impairment = impairment
        self.center_freq = center_freq
        self.carrier_spacing = carrier_spacing
        # position of transceiver/antenna
        self.cord_x = cord_x
        self.cord_y = cord_y
        self.cord_z = cord_z
        self.tx_ant_gain_db = 0
        self.rx_ant_gain_db = 0

    def set_position(self, cord_x, cord_y, cord_z):
        self.cord_x = cord_x
        self.cord_y = cord_y
        self.cord_z = cord_z

    def set_ant_gains(self, tx_ant_gain_db, rx_ant_gain_db):
        self.tx_ant_gain_db = tx_ant_gain_db
        self.rx_ant_gain_db = rx_ant_gain_db

    def transmit(self, in_bits, skip_dist=False, return_both=False):
        clean_symb = self.modem.modulate(in_bits)

        if skip_dist or self.impairment is None:
            return clean_symb
        elif return_both and self.impairment is not None:
            return self.impairment.process(clean_symb), clean_symb
        elif self.impairment is not None:
            return self.impairment.process(clean_symb)

    def receive(self, in_symb):
        return self.modem.demodulate(in_symb)
