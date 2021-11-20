from speedup import jit
import numpy as np
import scipy as scp
import matplotlib.pyplot as plt
import modulation


class LinearArray:
    def __init__(self, n_elements, base_modems, base_impairments, center_freq, wav_len_spacing):
        self.n_elements = n_elements
        self.base_modems = base_modems
        self.base_impariments = base_impairments
        self.center_freq = center_freq
        self.wav_len_spacing = wav_len_spacing
        self.array_elements = []

    def create_array(self):
        if isinstance(self.base_modems, list) and len(self.base_modems) == self.n_elements and len(self.base_impariments) == self.n_elements:
            for idx, base_modem in enumerate(self.base_modems):
                pass

        else:
            for idx in range(self.n_elements):
                pass



