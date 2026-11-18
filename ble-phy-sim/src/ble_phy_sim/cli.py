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


def cic_decimator(
    iq_in: NDArray[np.complex64],
    stages: int,
    r: int,
    n: int,
    pass_band: float,
    taps: int,
) -> NDArray[np.complex64]:
    # Integrator
    iq_out = iq_in

    for _ in range(stages):
        iq_out = sig.lfilter([1.0], [1.0, -1.0], iq_out)
    pass

    # Downsample
    iq_out = iq_out[::r]

    # Comb
    b_comb = np.zeros(n + 1, dtype=np.float64)
    b_comb[0] = 1.0
    b_comb[-1] = -1.0
    for _ in range(stages):
        iq_out = sig.lfilter(b_comb, [1.0], iq_out)

    # DC gain compensation
    dc_gain = (r * n) ** stages
    iq_out = iq_out / dc_gain

    # Compensation FIR filter
    fp = np.linspace(0, pass_band, 100)
    gp = (np.sinc(fp / r) / np.sinc(fp * n)) ** stages
    fs = np.linspace(min(pass_band + 0.1, 0.45), 0.5, 100)
    gs = np.zeros_like(fs)

    f = np.concatenate([fp, fs])
    g = np.concatenate([gp, gs])
    fir = sig.firwin2(taps, f, g, fs=1.0)

    iq_out = sig.lfilter(fir, [1.0], iq_out)

    return iq_out


def gfsk_demodulator(iq_in: NDArray[np.complex64]) -> NDArray[np.float64]:
    iq_delay = sig.lfilter([0, 1], [1], iq_in.astype(np.complex128))
    delay_conj_mult = iq_in * np.conj(iq_delay)
    return np.atan2(delay_conj_mult.imag, delay_conj_mult.real)


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

    # ==========================================================================
    # BASEBAND
    # ==========================================================================

    # Decimate to 4 MSPS. Passband @ 1 MHz
    decimated_iq = cic_decimator(iq_max2830, 4, 5, 2, 0.25, 31)
    decimated_iq = decimated_iq[0 : round(len(decimated_iq) / 4) * 4]

    # Demodulate
    demod = gfsk_demodulator(decimated_iq).clip(-0.3 * np.pi, 0.3 * np.pi)

    # I/Q magnitude
    magnitude = np.abs(decimated_iq)
    magnitude_avg = sig.lfilter(np.ones(32 * 4) / 32 / 4, [1], magnitude)
    magnitude_ok = 20 * np.log10(magnitude_avg) > -80

    # Detect CFO from preamble
    cfo_est_len = 6 * 4
    cfo = sig.lfilter(np.ones(cfo_est_len) / cfo_est_len, [1], demod)
    delay = np.zeros(32 * 4 + 1)
    delay[-1] = 1.0
    cfo = sig.lfilter(delay, [1], cfo)

    # Access address correlation
    access_addr = 0x8E89BED6
    aa_bits = np.array([int(bit) for bit in f"{access_addr:032b}"])
    aa_bits = aa_bits * 2 - 1
    aa_corr = sig.lfilter(aa_bits.repeat(4), [1], demod)
    aa_avg = sig.lfilter(np.ones_like(aa_bits.repeat(4)), [1], demod) / 32 / 4
    aa_corr_dc_free = aa_corr - aa_avg * np.sum(aa_bits)

    # Matched signal
    aa_match = (aa_corr_dc_free > 32) * magnitude_ok
    cfo_match = cfo * aa_match

    # Isolate frames
    match_locations = []
    cfos = []
    for i, match in enumerate(aa_match):
        if match:
            match_locations.append(i - (32 + 6) * 4)
            cfos.append(cfo_match[i])

    # Choose frame
    match_location = match_locations[0]
    cfo = cfos[0]

    # Recover symbol timing (Gardner Loop)
    detect = demod[match_location:] - cfo
    detect_delay_2 = sig.lfilter([0, 0, 1], [1], detect)
    detect_delay_4 = sig.lfilter([0, 0, 0, 0, 1], [1], detect)

    kp = 0.125
    phase = 0
    phase_error_acc = 0
    phase_error_accs = np.zeros_like(detect)
    samples = np.zeros_like(detect)
    corrected = False
    for i, _ in enumerate(detect):
        if i % 4 == phase % 4 and not corrected:
            error = detect_delay_2[i] * (detect[i] - detect_delay_4[i])
            phase_error_acc += error * kp
            phase_error_accs[i] = phase_error_acc
            samples[i] = 0.3
        if phase_error_acc > 0.25:
            phase_error_acc -= 0.25
            phase -= 1
            corrected = True
        elif phase_error_acc < -0.25:
            phase_error_acc += 0.25
            phase += 1
            corrected = True
        else:
            corrected = False

    # Bit detector
    # Synchronize frame with access address
    # Obtain PDU & CRC

    # ==========================================================================
    # PLOTTING
    # ==========================================================================

    # FFT
    input_fft = np.fft.fft(iq) / len(iq)
    input_freqs = np.fft.fftfreq(len(input_fft), 1 / 100e6) + fc_in
    max2830_fft = np.fft.fft(iq_max2830) / len(iq_max2830)
    max2830_freqs = np.fft.fftfreq(len(max2830_fft), 1 / 20e6)
    decimated_fft = np.fft.fft(decimated_iq) / len(decimated_iq)
    decimated_freqs = np.fft.fftfreq(len(decimated_fft), 1 / 4e6)

    # Setup plot
    plt.style.use("dark_background")
    layout = [
        ["A", "A"],
        ["B", "C"],
        ["D", "D"],
        ["E", "E"],
    ]
    fig, axs = plt.subplot_mosaic(layout)
    fig.tight_layout()

    # Input specturm
    axs["A"].plot(input_freqs, 20 * np.log10(np.abs(input_fft)))
    axs["A"].plot(max2830_freqs, 20 * np.log10(np.abs(max2830_fft)))
    axs["A"].plot(decimated_freqs, 20 * np.log10(np.abs(decimated_fft)))
    axs["A"].xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    axs["A"].yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    axs["A"].set_xlabel("Frequency (Hz)")
    axs["A"].set_ylabel("Amplitude (dB)")
    axs["A"].set_title(f"Input I/Q FFT (Fs = {fs_in / 1e6} MSPS)")
    axs["A"].set_ylim(-150, -80)
    axs["A"].set_xticks(np.arange(2400e6, 2482e6, 2e6))
    axs["A"].set_xlim(2401e6, 2481e6)
    axs["A"].grid()

    # Channel spectrum
    axs["B"].plot(
        input_freqs - lo_f,
        20 * np.log10(np.abs(input_fft)),
        label="Input spectrum",
    )
    axs["B"].plot(
        max2830_freqs,
        20 * np.log10(np.abs(max2830_fft)),
        label="MAX2830 spectrum",
    )
    axs["B"].plot(
        decimated_freqs,
        20 * np.log10(np.abs(decimated_fft)),
        label="4 MSPS spectrum",
    )
    axs["B"].xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    axs["B"].yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    axs["B"].set_xlabel("Frequency (Hz)")
    axs["B"].set_ylabel("Amplitude (dB)")
    axs["B"].set_title(
        f"Input I/Q FFT (Fc = {lo_f / 1e6} MHz, Fs = {fs_in / 1e6} MSPS, 20 MSPS, 4 MSPS)"
    )
    axs["B"].set_ylim(-150, -80)
    axs["B"].set_xticks(np.arange(-2e6, 2e6, 500e3))
    axs["B"].set_xlim(-2e6, 2e6)
    axs["B"].grid()
    axs["B"].legend()

    # IQ plot
    axs["C"].set_title("I/Q plot")
    axs["C"].plot(iq_max2830.real, iq_max2830.imag, label="MAX2830 spectrum")
    axs["C"].plot(decimated_iq.real, decimated_iq.imag, label="4 MSPS spectrum")
    axs["C"].legend()
    axs["C"].grid()

    # Demod plot
    axs["D"].set_title("Demodulated plot")
    axs["D"].plot(np.arange(len(demod)), demod, label="Demodulated signal")
    axs["D"].plot(
        np.arange(len(aa_match)), aa_match, label="Access Address Match"
    )
    axs["D"].plot(np.arange(len(cfo_match)), cfo_match, label="CFO @ match")
    axs["D"].plot(
        np.arange(len(magnitude_ok)), magnitude_ok, label="Magnitude OK"
    )
    axs["D"].set_ylim(-0.5, 0.5)
    axs["D"].grid()
    axs["D"].legend()

    # Frame plot
    axs["E"].set_title("Frame plot")
    axs["E"].plot(np.arange(len(detect)), detect, label="Detect")
    axs["E"].plot(
        np.arange(len(phase_error_accs)),
        phase_error_accs,
        label="Phase accumulator",
    )
    axs["E"].stem(np.arange(len(samples)), samples, label="samples")
    axs["E"].set_ylim(-0.5, 0.5)
    axs["E"].set_xlim(0, 128 * 4)
    axs["E"].grid()
    axs["E"].legend()

    # Plot
    plt.show()
