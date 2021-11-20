from speedup import jit
import numpy as np
import scipy as scp
import matplotlib.pyplot as plt
import modulation
import impairments


class Transceiver:
    def __init__(self, modem, impairment=None):

        self.modem = modem
        self.impairment = impairment

    def transmit(self, in_bits, skip_dist=False):
        tx_symb = self.modem.modulate(in_bits)

        if self.impairment is not None and skip_dist is not True:
            return self.impairment.process(tx_symb)
        else:
            return tx_symb

    def receive(self, in_symb):
        return self.modem.demodulate(in_symb)
