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


def max2830_demod(
    iq_in: NDArray[np.complex64],
    fc_in: float,
    fs_in: float,
    f_lo: float,
) -> NDArray[np.complex64]:
    # Shift to target channel frequency
    delta_freq = f_lo - fc_in
    t = np.arange(len(iq_in)) / fs_in
    iq_shifted = iq_in.astype(np.complex128) * np.exp(
        -1j * 2 * np.pi * delta_freq * t
    )

    # MAX2830 baseband LPF
    cutoff = 7.5e6
    sos_lpf = sig.butter(6, cutoff, btype="lowpass", fs=fs_in, output="sos")

    # MAX2830 baseband HPF
    sos_hpf = sig.butter(1, 100, btype="highpass", fs=fs_in, output="sos")

    # MAX2830 response
    sos_max2830 = np.vstack((sos_lpf, sos_hpf))

    # Filter & resample
    iq_filtered = sig.sosfilt(sos_max2830, iq_shifted)

    # Decimate to 20 MSPS
    r = int(fs_in / 20e6)
    return iq_filtered[::r].astype(np.complex64)


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: ble_phy_sim <data_file_path>")
        sys.exit(1)

    data_file_path = sys.argv[1]

    # Original IQ data
    dtype = np.dtype(np.complex64)
    fs_in = 100e6
    fc_in = 2441.5e6
    iq = np.fromfile(data_file_path, dtype=dtype)

    # Get channel frequency
    ch_idx = 39
    lo_f = get_ble_channel_freq(ch_idx)

    # Get IQ data from MAX2830
    iq_max2830 = max2830_demod(iq, fc_in, fs_in, lo_f)

    # FFT
    input_fft = np.fft.fft(iq)
    input_freqs = np.fft.fftfreq(len(input_fft), 1 / 100e6) + fc_in
    max2830_fft = np.fft.fft(iq_max2830)
    max2830_freqs = np.fft.fftfreq(len(max2830_fft), 1 / 20e6)

    # Setup plot
    _, axs = plt.subplots(2, 1)

    # 100 MSPS data
    axs[0].plot(input_freqs, 20 * np.log10(np.abs(input_fft)))
    axs[0].xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    axs[0].yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    axs[0].set_xlabel("Frequency (Hz)")
    axs[0].set_ylabel("Amplitude (dB)")
    axs[0].set_title(
        f"Input I/Q FFT (Fc = {fc_in / 1e6} MHz, Fs = {fs_in / 1e6} MSPS)"
    )
    axs[0].grid()
    axs[0].set_ylim(-60, 30)
    axs[0].set_xticks(np.arange(2400e6, 2482e6, 2e6))
    axs[0].set_xlim(2401e6, 2481e6)

    # MAX2830 data
    axs[1].plot(max2830_freqs, 20 * np.log10(np.abs(max2830_fft)))
    axs[1].xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    axs[1].yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    axs[1].set_xlabel("Frequency (Hz)")
    axs[1].set_ylabel("Amplitude (dB)")
    axs[1].set_title(f"MAX2830 I/Q FFT (LO = {lo_f / 1e6} MHz, Fs = 20 MSPS)")
    axs[1].grid()
    axs[1].set_ylim(-60, 30)
    axs[1].set_xticks(np.arange(-10e6, 10e6, 2e6))
    axs[1].set_xlim(-10e6, 10e6)

    # Plot
    plt.show()
