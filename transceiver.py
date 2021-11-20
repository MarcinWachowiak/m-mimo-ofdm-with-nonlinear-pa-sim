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

    def transmit(self, in_bits, skip_dist=False, return_both=False):
        clean_symb = self.modem.modulate(in_bits)

        if skip_dist:
            return clean_symb
        elif return_both and self.impairment is not None:
            return self.impairment.process(clean_symb), clean_symb
        elif self.impairment is not None:
            return self.impairment.process(clean_symb)

    def receive(self, in_symb):
        return self.modem.demodulate(in_symb)
