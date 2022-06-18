import csv
import numpy as np
from datetime import datetime
vectorized_binary_repr = np.vectorize(np.binary_repr)

from speedup import jit
import matplotlib.pyplot as plt
from matplotlib import colors
import torch


# TODO: Inspect faster ways of dec to bin, bin to dec conversion
# TODO: Add code documentation


def dec2bitarray(in_number, bit_width):
    """
    Converts a positive integer or an array-like of positive integers to NumPy array of the specified size containing
    bits (0 and 1).

    Parameters
    ----------
    in_number : int or array-like of int
        Positive integer to be converted to a bit array.

    bit_width : int
        Size of the output bit array.

    Returns
    -------
    bitarray : 1D ndarray of numpy.int8
        Array containing the binary representation of all the input decimal(s).

    """

    if isinstance(in_number, (np.integer, int)):
        return decimal2bitarray(in_number, bit_width).copy()
    result = np.zeros(bit_width * len(in_number), np.int8)
    for pox, number in enumerate(in_number):
        result[pox * bit_width:(pox + 1) * bit_width] = decimal2bitarray(number, bit_width).copy()
    return result


def decimal2bitarray(number, bit_width):
    """
    Converts a positive integer to NumPy array of the specified size containing bits (0 and 1). This version is slightly
    quicker that dec2bitarray but only work for one integer.

    Parameters
    ----------
    in_number : int
        Positive integer to be converted to a bit array.

    bit_width : int
        Size of the output bit array.

    Returns
    -------
    bitarray : 1D ndarray of numpy.int8
        Array containing the binary representation of all the input decimal(s).

    """
    result = np.zeros(bit_width, np.int8)
    i = 1
    pox = 0
    while i <= number:
        if i & number:
            result[bit_width - pox - 1] = 1
        i <<= 1
        pox += 1
    return result


def bitarray2dec(in_bitarray):
    """
    Converts an input NumPy array of bits (0 and 1) to a decimal integer.

    Parameters
    ----------
    in_bitarray : 1D ndarray of ints
        Input NumPy array of bits.

    Returns
    -------
    number : int
        Integer representation of input bit array.
    """

    number = 0

    for i in range(len(in_bitarray)):
        number = number + in_bitarray[i] * pow(2, len(in_bitarray) - 1 - i)

    return number


def hamming_dist(in_bitarray_1, in_bitarray_2):
    """
    Computes the Hamming distance between two NumPy arrays of bits (0 and 1).

    Parameters
    ----------
    in_bit_array_1 : 1D ndarray of ints
        NumPy array of bits.

    in_bit_array_2 : 1D ndarray of ints
        NumPy array of bits.

    Returns
    -------
    distance : int
        Hamming distance between input bit arrays.
    """

    distance = np.bitwise_xor(in_bitarray_1, in_bitarray_2).sum()

    return distance


def euclid_dist(in_array1, in_array2):
    """
    Computes the squared euclidean distance between two NumPy arrays

    Parameters
    ----------
    in_array1 : 1D ndarray of floats
        NumPy array of real values.

    in_array2 : 1D ndarray of floats
        NumPy array of real values.

    Returns
    -------
    distance : float
        Squared Euclidean distance between two input arrays.
    """
    distance = ((in_array1 - in_array2) * (in_array1 - in_array2)).sum()

    return distance


def upsample(x, n):
    """
    Upsample the input array by a factor of n

    Adds n-1 zeros between consecutive samples of x

    Parameters
    ----------
    x : 1D ndarray
        Input array.

    n : int
        Upsampling factor

    Returns
    -------
    y : 1D ndarray
        Output upsampled array.
    """
    y = np.empty(len(x) * n, dtype=complex)
    y[0::n] = x
    zero_array = np.zeros(len(x), dtype=complex)
    for i in range(1, n):
        y[i::n] = zero_array

    return y


@jit(nopython=True)
def td_signal_power(signal):
    sig_power = np.mean(np.abs(signal) ** 2)
    return sig_power


@jit(nopython=True)
def fd_signal_power(signal):
    sig_power = np.sum(np.abs(signal) ** 2)
    return sig_power


@jit(nopython=True)
def count_mismatched_bits(tx_bits_arr, rx_bits_arr):
    return np.bitwise_xor(tx_bits_arr, rx_bits_arr).sum()


@jit(nopython=True)
def ebn0_to_snr(eb_per_n0, n_fft, n_sub_carr, constel_size):
    return 10 * np.log10(10 ** (eb_per_n0 / 10) * n_sub_carr * np.log2(constel_size) / n_fft)


@jit(nopython=True)
def snr_to_ebn0(snr, n_fft, n_sub_carr, constel_size):
    return 10 * np.log10(10 ** (snr / 10) * (n_fft / (n_sub_carr * np.log2(constel_size))))


@jit(nopython=True)
def to_db(samples):
    return 10 * np.log10(samples)


# points start from X=r Y=0 and then proceed anticlockwise
def pts_on_circum(r, n=100):
    return [(np.cos(2 * np.pi / n * x) * r, np.sin(2 * np.pi / n * x) * r) for x in range(0, n + 1)]


def pts_on_semicircum(r, n=100):
    return [(np.cos(np.pi / n * x) * r, np.sin(np.pi / n * x) * r) for x in range(0, n + 1)]


def plot_spatial_config(ant_array, rx_transceiver, plot_3d=True):
    if plot_3d:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        tx_cord_x = []
        tx_cord_y = []
        tx_cord_z = []
        for transceiver in ant_array.array_elements:
            tx_cord_x.append(transceiver.cord_x)
            tx_cord_y.append(transceiver.cord_y)
            tx_cord_z.append(transceiver.cord_z)

        # plot line form array center to rx
        ax.plot([ant_array.cord_x, rx_transceiver.cord_x], [ant_array.cord_y, rx_transceiver.cord_y],
                [ant_array.cord_z, rx_transceiver.cord_z], color="C2",
                linestyle='--', label="LOS")

        ax.scatter(tx_cord_x, tx_cord_y, tx_cord_z, color="C0", marker='^', label="TX")
        ax.scatter(rx_transceiver.cord_x, rx_transceiver.cord_y, rx_transceiver.cord_z, color="C1", marker='o',
                   label="RX")

        # color ground surface
        ax.zaxis.set_pane_color(colors.to_rgba("gray"))

        ax.set_title('TX RX spatial configuration')
        ax.set_xlabel("X plane [m]")
        ax.set_ylabel("Y plane [m]")
        ax.set_zlabel("Z plane [m]")
        ax.legend()
        ax.grid()
        ax.set_axisbelow(True)
        plt.tight_layout()
        plt.savefig("figs/spatial_rx_tx_config_3d.png", dpi=600, bbox_inches='tight')
        plt.show()
    else:
        fig, ax = plt.subplots()
        tx_cord_x = []
        tx_cord_y = []
        # plot line form array center to rx
        ax.plot([ant_array.cord_x, rx_transceiver.cord_x], [ant_array.cord_y, rx_transceiver.cord_y], color="C2",
                linestyle='--')

        for transceiver in ant_array.array_elements:
            tx_cord_x.append(transceiver.cord_x)
            tx_cord_y.append(transceiver.cord_y)

        ax.scatter(tx_cord_x, tx_cord_y, color="C0", marker='^', label="TX")
        ax.scatter(rx_transceiver.cord_x, rx_transceiver.cord_y, color="C1", marker='o', label="RX")

        ax.plot_surface()
        ax.set_aspect('equal', 'box')
        ax.set_title('Spatial configuration TX - RX')
        ax.set_xlabel("X plane [m]")
        ax.set_ylabel("Y plane [m]")
        ax.grid()
        ax.legend()
        ax.set_axisbelow(True)
        plt.tight_layout()
        plt.savefig("figs/spatial_rx_tx_config_2d.png", dpi=600, bbox_inches='tight')
        plt.show()


def plot_array_config(ant_array, plot_3d=False):
    if plot_3d:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        for transceiver in ant_array.array_elements:
            ax.scatter(transceiver.cord_x, transceiver.cord_y, transceiver.cord_z, color="C0", marker='^')

        ax.set_title('Antenna array')
        ax.set_xlabel("X plane [m]")
        ax.set_ylabel("Y plane [m]")
        ax.set_zlabel("Z plane [m]")
        ax.grid()
        ax.set_axisbelow(True)
        plt.show()
    else:
        fig, ax = plt.subplots()
        for transceiver in ant_array.array_elements:
            ax.scatter(transceiver.cord_x, transceiver.cord_y, color="C0", marker='^')
        ax.set_title('Antenna array')
        ax.set_xlabel("X plane [m]")
        ax.set_ylabel("Y plane [m]")
        ax.grid()
        ax.set_axisbelow(True)
        plt.tight_layout()
        plt.show()


def to_freq_domain(in_sig_td, remove_cp=True, cp_len=None):
    # remove cp from in sig matrix
    if remove_cp:
        if in_sig_td.ndim == 1:
            in_sig = in_sig_td[cp_len:]
        else:
            in_sig = in_sig_td[:, cp_len:]
    else:
        in_sig = in_sig_td
    # perform fft row wise
    return torch.fft.fft(torch.from_numpy(in_sig), norm="ortho").numpy()


def to_time_domain(in_sig_mat_fd):
    return torch.fft.ifft(torch.from_numpy(in_sig_mat_fd), norm="ortho").numpy()


def save_to_csv(data_lst, filename):
    timestamp = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
    with open("../figs/csv_results/%s_%s.csv" % (filename, timestamp), 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerows(data_lst)

def read_from_csv(filename):
    with open("../figs/csv_results/%s" % filename, 'r', newline='') as csv_file:
        reader = csv.reader(csv_file)
        data_lst = list(reader)
    return data_lst