import csv
import sys

import numpy as np

vectorized_binary_repr = np.vectorize(np.binary_repr)

from speedup import jit
import matplotlib.pyplot as plt
from matplotlib import colors
import torch

from numpy import ndarray


# TODO: Inspect the performance of dec2bin bin2dec and find faster ways to do it

def dec2bitarray(in_number: ndarray, bit_width: int) -> ndarray:
    """
    Convert the input data into bits.

    :param in_number: int or array of ints
    :param bit_width: size of the output bit array
    :return: binary representation of the input - array of bits
    """

    if isinstance(in_number, (np.integer, int)):
        return decimal2bitarray(in_number, bit_width).copy()
    result = np.zeros(bit_width * len(in_number), np.int8)
    for pox, number in enumerate(in_number):
        result[pox * bit_width:(pox + 1) * bit_width] = decimal2bitarray(number, bit_width).copy()
    return result


def decimal2bitarray(number: int, bit_width: int) -> ndarray:
    """
    Convert the integer into bit array.

    :param number: input integer
    :param bit_width: size of the output bit array (size of the input in bits)
    :return: binary representation of the input - array of bits
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


def bitarray2dec(in_bitarray: ndarray) -> int:
    """
    Convert bitarray to the decimal representation.

    :param in_bitarray: input array of bits
    :return: integer representation of the input
    """

    number = 0

    for i in range(len(in_bitarray)):
        number = number + in_bitarray[i] * pow(2, len(in_bitarray) - 1 - i)

    return number


@jit(nopython=True)
def td_signal_power(signal: ndarray) -> float:
    """
    Calculate the power of the signal in the time domain.

    :param signal: input signal vector
    :return: calculated power value
    """
    sig_power = np.mean(np.abs(signal) ** 2)
    return sig_power


@jit(nopython=True)
def fd_signal_power(signal: ndarray) -> float:
    """
    Calculate the power of the signal in the frequency domain.

    :param signal: input signal vector
    :return: calculated power value
    """
    sig_power = np.sum(np.abs(signal) ** 2)
    return sig_power


@jit(nopython=True)
def count_mismatched_bits(tx_bits_arr: ndarray, rx_bits_arr: ndarray) -> int:
    """
    Calculate the number of bit errors between two vectors.

    :param tx_bits_arr: transmitted bit array - reference
    :param rx_bits_arr: received bit array
    :return: number of bit errors
    """

    return np.bitwise_xor(tx_bits_arr, rx_bits_arr).sum()


@jit(nopython=True)
def ebn0_to_snr(eb_per_n0, n_fft: int, n_sub_carr: int, constel_size: int):
    """
    Convert the Eb/N0 to SNR.

    :param eb_per_n0: energy per bit to noise power spectral density ratio in [dB]
    :param n_fft: size of the inverse Fourier Transform - total number of subcarriers
    :param n_sub_carr: number of data subcarriers
    :param constel_size: size of the constellation
    :return: SNR value in [dB]
    """
    return 10 * np.log10(10 ** (eb_per_n0 / 10) * n_sub_carr * np.log2(constel_size) / n_fft)


@jit(nopython=True)
def snr_to_ebn0(snr, n_fft: int, n_sub_carr: int, constel_size: int):
    """
    Convert the SNR to Eb/N0.

    :param snr: signal-to-noise ratio in [dB]
    :param n_fft: size of the inverse Fourier Transform - total number of subcarriers
    :param n_sub_carr: number of data subcarriers
    :param constel_size: size of the constellation
    :return: SNR value in [dB]
    :return: Eb/N0 value in [dB]
    """
    return 10 * np.log10(10 ** (snr / 10) * (n_fft / (n_sub_carr * np.log2(constel_size))))


def to_db(samples):
    """
    Convert the values to the decibel scale [dB]
    :param samples: input data vector
    :return: data
    """
    return 10 * np.log10(samples)


# points start from X=r Y=0 and then proceed anticlockwise
def pts_on_circum(radius: float, n_points: int = 100) -> list:
    """
    Generate a number uniformly spaced points on a circumference.

    :param radius: radius of the circle
    :param n_points: number of equally spaced points
    :return: vector tuples containing the coordinates of the points (x,y)
    """
    return [(np.cos(2 * np.pi / n_points * x) * radius, np.sin(2 * np.pi / n_points * x) * radius) for x in
            range(0, n_points + 1)]


def pts_on_semicircum(radius: float, n_points: int = 100) -> list:
    """
    Generate a number uniformly spaced points on a semi circumference.

    :param radius: radius of the circle
    :param n_points: number of equally spaced points
    :return: vector tuples containing the coordinates of the points (x,y)
    """
    return [(np.cos(np.pi / n_points * x) * radius, np.sin(np.pi / n_points * x) * radius) for x in
            range(0, n_points + 1)]


def pts_on_semisphere(radius: float, n_points: int = 100, center_x: float = 0, center_y: float = 0,
                      center_z: float = 0):
    """
    Generate a number of points uniformly spaced on a semi sphere.

    :param radius: radius of the circle
    :param n_points: number of equally spaced points in azimuth and elevation (per row and column)
    :param center_x: X coordinate of the center of the sphere [m]
    :param center_y: Y coordinate of the center of the sphere [m]
    :param center_z: Z coordinate of the center of the sphere [m]
    :return: vector of tuples containing the coordinates of the points (x,y,z)
    """

    azimuth_angle_vec = np.deg2rad(np.linspace(0, 180, int(np.sqrt(n_points)), endpoint=True))
    elevation_angle_vec = np.deg2rad(np.linspace(0, 180, int(np.sqrt(n_points)), endpoint=True))
    rx_points_lst = []
    for azimuth_angle in azimuth_angle_vec:
        for elevation_angle in elevation_angle_vec:
            rx_pos_x = -radius * np.sin(elevation_angle) * np.cos(azimuth_angle) + center_x
            rx_pos_y = -radius * np.sin(elevation_angle) * np.sin(azimuth_angle) + center_y
            rx_pos_z = -radius * np.cos(elevation_angle) + center_z
            rx_points_lst.append((rx_pos_x, rx_pos_y, rx_pos_z))
    return rx_points_lst


def plot_spatial_config(ant_array, rx_transceiver=None, rx_points_lst: list = None, plot_3d: bool = True) -> None:
    """
    Plot the spatial configuration of the antenna array and the receiver.

    :param ant_array: antenna array object
    :param rx_transceiver: receiver object
    :param rx_points_lst: receiver points list
    :param plot_3d: flag if to plot in 3D
    :return: None
    """
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
        ax.scatter(tx_cord_x, tx_cord_y, tx_cord_z, color="C0", marker='^', label="TX")
        if rx_transceiver is not None:
            # plot line form array center to rx
            ax.plot([ant_array.cord_x, rx_transceiver.cord_x], [ant_array.cord_y, rx_transceiver.cord_y],
                    [ant_array.cord_z, rx_transceiver.cord_z], color="C2",
                    linestyle='--', label="LOS")
            ax.scatter(rx_transceiver.cord_x, rx_transceiver.cord_y, rx_transceiver.cord_z, color="C1", marker='o',
                       label="RX")
        elif rx_points_lst is not None:
            rx_cord_x = []
            rx_cord_y = []
            rx_cord_z = []
            for rx_point in rx_points_lst:
                rx_cord_x.append(rx_point[0])
                rx_cord_y.append(rx_point[1])
                rx_cord_z.append(rx_point[2])
            ax.scatter(rx_cord_x, rx_cord_y, rx_cord_z, color="C1", marker='o',
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
        plt.savefig("../figs/msc_figs/spatial_rx_tx_config_3d.png", dpi=600, bbox_inches='tight')
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
        plt.savefig("../figs/msc_figs/spatial_rx_tx_config_2d.png", dpi=600, bbox_inches='tight')
        plt.show()


def plot_array_config(ant_array, plot_3d: bool = False) -> None:
    """
    Plot the architecture/configuration of the antenna array.

    :param ant_array: antenna array object
    :param plot_3d: flag if to plot in 3D
    :return: None
    """

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


def to_freq_domain(in_sig_td: ndarray, remove_cp: bool = True, cp_len: int = None) -> ndarray:
    """
    Convert the input signal to frequency domain.

    :param in_sig_td: input signal vector in time domain
    :param remove_cp: flag if to remove the cyclic prefixs
    :param cp_len: cyclic prefix length
    :return: signal vector in frequency domain
    """
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


def to_time_domain(in_sig_mat_fd: ndarray) -> ndarray:
    """
    Convert the input signal to time domain.

    :param in_sig_mat_fd: input signal vector in frequency domain.
    :return: signal vector in time domain
    """
    return torch.fft.ifft(torch.from_numpy(in_sig_mat_fd), norm="ortho").numpy()


def save_to_csv(data_lst: list, filename: str) -> None:
    """
    Save the data list to comma-separated values (CSV) file.

    :param data_lst: list of vectors to save (the data vectors have to be flat - no nesting inside them)
    :param filename: filename of the CSV file
    :return: None
    """
    with open("figs/csv_results/%s.csv" % filename, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerows(data_lst)


def read_from_csv(filename: str) -> list:
    """
    Read the data from the comma-separated values (CSV) file.

    :param filename: filename of the CSV file
    :return: list of read data vectors
    """
    with open("../figs/csv_results/%s.csv" % filename, 'r', newline='') as csv_file:
        reader = csv.reader(csv_file, quoting=csv.QUOTE_NONNUMERIC)
        data_lst = list(reader)
    return data_lst


# Print iterations progress
def print_progress_bar(iteration: int, total: int, prefix: str = '', suffix: str = '', decimals: int = 1,
                       bar_length: int = 100) -> None:
    """
    Creates a progress bar in terminal.

    :param iteration: number of the current iteration
    :param total: total number of iterations
    :param prefix: prefix string
    :param suffix: suffix string
    :param decimals: number of decimals in percent complete
    :param bar_length: character length of the bar
    :return: None
    """

    str_format = "{0:." + str(decimals) + "f}"
    percents = str_format.format(100 * (iteration / float(total)))
    filled_length = int(round(bar_length * iteration / float(total)))
    bar = '|' * filled_length + '-' * (bar_length - filled_length)

    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix)),

    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()
