from speedup import jit
import numpy as np
import scipy as scp
import matplotlib.pyplot as plt
import modulation


class Transceiver:
    def __init__(self, modem, impairment):

        self.modem = modem
        self.impairment = impairment




