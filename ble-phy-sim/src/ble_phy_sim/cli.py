import sys

import numpy as np
from numpy.typing import NDArray

import scipy.signal as sig

import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter


def get_ble_channel_freq(channel_idx: int) -> float:
    if channel_idx == 37:
        return 2402e6
    elif channel_idx <= 10:
        return 2404e6 + 2e6 * channel_idx
    elif channel_idx == 38:
        return 2426e6
    elif channel_idx <= 36:
        return 2428e6 + 2e6 * (channel_idx - 11)
    else:
        return 2480e6


def get_channel_iq_data(
    data: NDArray[np.complex64],
    center_frequency: float,
    sample_rate: float,
    channel_idx: int,
) -> NDArray[np.complex64]:
    delta_freq = get_ble_channel_freq(channel_idx) - center_frequency
    t = np.arange(len(data)) / sample_rate
    return data * np.exp(-1j * 2 * np.pi * delta_freq * t)


def decimate(
    data: NDArray[np.complex64], sample_rate: float, target_sample_rate: float
) -> NDArray[np.complex64]:
    # Create FIR filter
    num_taps = 1001
    cutoff = target_sample_rate / 2 * 0.8
    fir_taps = sig.firwin(num_taps, cutoff, fs=sample_rate)

    # Filter & resample
    resampled_data = sig.resample_poly(
        data, 1, sample_rate / target_sample_rate, window=fir_taps
    )

    return resampled_data


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: ble_phy_sim <data_file_path>")
        sys.exit(1)

    data_file_path = sys.argv[1]

    # Data
    dtype = np.dtype(np.complex64)
    sample_rate = 100e6
    center_frequency = 2441.5e6
    data = np.fromfile(data_file_path, dtype=dtype)

    # Get channel data
    target_channel_idx = 39
    target_center_frequency = get_ble_channel_freq(target_channel_idx)
    channel_data = get_channel_iq_data(
        data, center_frequency, sample_rate, target_channel_idx
    )

    # Decimate
    target_sample_rate = 2e6
    decimated_channel_data = decimate(
        channel_data, sample_rate, target_sample_rate
    )

    # Retarget
    center_frequency = target_center_frequency
    sample_rate = target_sample_rate
    data = decimated_channel_data

    # FFT
    fft_result = np.fft.fft(data)
    freq_axis = np.fft.fftfreq(len(fft_result), 1 / sample_rate)
    freq_axis_shifted = freq_axis + center_frequency

    # Setup plot
    _, ax = plt.subplots(figsize=(12, 5))
    ax.plot(freq_axis_shifted, 20 * np.log10(np.abs(fft_result)))
    ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Amplitude (dB)")
    ax.grid()
    ax.set_title("FFT")
    ax.set_ylim(-60)

    # Plot
    plt.show()
