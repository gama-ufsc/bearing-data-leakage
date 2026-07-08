"""
Código base para o cálculo do espectrograma e da curtose espectral de um sinal.
"""

import numpy as np
import scipy as sp
from scipy.fft import fft, fftfreq


def get_fft(data, n=None, return_onesided=True, remove_mean=True, reduce_size=False):
    """
    Compute the one-dimensional discrete Fourier transform

    Parameters
    ----------
    data : array_like
        Input data.
    n : int, opt
        Number of samples.
    return_onesided : bool, opt
        If True, returns a one-sided spectrum (valid for real data).
    remove_mean : bool, opt
        If True, removes the average signal
    reduce_size : bool, opt
        If True, rounds FFT results to a given number of decimals

    Returns
    -------
    np.ndarray
        Array of amplitudes of the discrete Fourier transform coeficients.
    """

    # Number of samples
    if n is None:
        n = int(len(data))
    elif n < int(len(data)):
        raise ValueError(
            "Number of samples should be greater than or equal to the original vector length"
        )

    # Remove average signal
    if remove_mean:
        data = data - np.mean(data)

    # FFT
    sp = fft(data, n=n) / n

    # Correction factor in case of zero padding
    sp = sp * n / len(data)

    # Return a one sided spectrum if desired
    if return_onesided:
        out = np.abs(sp[: n // 2])
        out[1:] = out[1:] * 2
    else:
        out = np.abs(sp)

    # Reduce memory usage for front-end
    if reduce_size:
        out = np.round(out, decimals=5)

    return out


def get_fftfreq(samples, fs, return_onesided=True):
    """
    Return the discrete Fourier transform sample frequencies

    Parameters
    ----------
    samples : int
        Number of samples.
    fs : int
        Sampling rate.
    return_onesided : bool, opt
        If True, returns a one-sided spectrum (valid for real data).

    Returns
    -------
    np.ndarray
        Array of frequencies
    """
    freq = fftfreq(samples, 1 / fs)
    if return_onesided:
        freq = freq[: samples // 2]
    return freq


def enbw(window, fs=None):
    """
    Return the equivalent noise bandwidth of a window.

    Parameters
    ----------
    window : sp.signal.get_window
        Scipy window
    fs : int, optional
        Sampling frequency, by default None

    Returns
    -------
    float
        Equivalent noise bandwidth
    """
    area = np.sum(np.abs(window) ** 2)
    height = np.abs(np.sum(window)) ** 2
    if fs is None:
        return area / height * len(window)
    else:
        return area / height * fs


def get_nperseg(samples):
    """
    Return the number of points per segment for the spectrogram.

    Parameters
    ----------
    samples : int
        Number of samples.

    Returns
    -------
    int
        Number of points per segment
    """
    if samples >= 2 and samples <= 63:
        divisor = 2
    elif samples >= 64 and samples <= 255:
        divisor = 4
    elif samples >= 256 and samples <= 2047:
        divisor = 8
    elif samples >= 2048 and samples <= 4095:
        divisor = 16
    elif samples >= 4096 and samples <= 8191:
        divisor = 32
    elif samples >= 8192 and samples <= 16383:
        divisor = 64
    elif samples >= 16384:
        divisor = 128

    nperseg = samples // divisor

    # print(f'Divisor: {divisor}')

    return nperseg


def get_noverlap(overlap, nperseg, window_name):
    """
    Return the number of points to overlap between segments for the spectrogram.

    Parameters
    ----------
    overlap : str, float
        Overlap mode if "Auto" or the percentage of overlap.
    nperseg : int
        Number of points per segment.
    window_name : str
        Scipy window name.

    Returns
    -------
    int
        Number of points to overlap between segments
    """
    if overlap == "Auto":
        bw = enbw(sp.signal.get_window(window=window_name, Nx=nperseg - 1))
        overlap = 1 - 1 / (2 * bw - 1)
    else:
        overlap = float(overlap) / 100

    noverlap = int(overlap * nperseg)

    # print('Overlap: {:.2f} %'.format(overlap * 100))

    return noverlap


# @curry
def get_spectrogram(
    x,
    fs,
    nperseg,
    noverlap,
    window_name,
    remove_zero_freq=True,
    remove_Nyquist_freq=True,
):
    """
    Return the spectrogram of a signal.

    Parameters
    ----------
    x : np.array
        Signal to be analyzed.
    fs : int
        Sampling frequency.
    nperseg : int
        Number of points per segment.
    noverlap : int
        Number of points to overlap between segments.
    window_name : str
        Window to be used in the STFT.
    remove_zero_freq : bool, optional
        If remove zero frequency, by default True
    remove_Nyquist_freq : bool, optional
        If remove nyquist freq, by default True

    Returns
    -------
    Sxx : np.array
        Spectrogram of the signal.
    f : np.array
        Frequency axis.
    t : np.array
        Time axis.
    """
    # Calculate the short-time Fourier Transform (STFT)
    f, t, Sxx = sp.signal.spectrogram(
        x=x,
        fs=fs,
        nperseg=nperseg,
        noverlap=noverlap,
        scaling="spectrum",
        window=window_name,
        mode="magnitude",
    )

    # remove Nyquist frequency
    if remove_Nyquist_freq:
        Sxx = np.delete(Sxx, -1, 0)
        f = np.delete(f, -1, 0)

    # remove zero frequency
    if remove_zero_freq:
        Sxx = np.delete(Sxx, 0, 0)
        f = np.delete(f, 0, 0)

    return Sxx, f, t


def get_spectral_kurtosis(Sxx):
    """
    Return the spectral kurtosis of a signal.

    Parameters
    ----------
    Sxx : float 2D array
        Spectrogram of a signal

    Returns
    -------
    SK : float 1D array
        Spectral kurtosis of the signal
    """

    return np.mean((Sxx) ** 4, axis=1) / (np.mean((Sxx) ** 2, axis=1) ** 2) - 2


if __name__ == "__main__":
    # @markdown ---
    samples = 1024
    fs = 2048
    sig = np.ones(samples)

    # @markdown Espectrograma configurações:
    # @markdown Insira a resolução no tempo em s (inverso da resolução na frequência em Hz). Deixe em Auto para uma escolha automática com base nos parâmetros do sinal
    time_resolution = "Auto"  # @param ['Auto'] {allow-input: true}
    # @markdown Insira a quantidade de overlap em %. Deixe em Auto para uma escolha automática com base nos parâmetros do sinal
    overlap = "Auto"  # @param ['Auto'] {allow-input:true}
    # @markdown Escolha o janelamento
    window_name = "Hann"  # @param ['Rectangular','Hamming','Hann','Bohman','4-term Blackman-Harris']

    if window_name == "Rectangular":
        window_name = "boxcar"
    elif window_name == "Hamming":
        window_name = "hamming"
    elif window_name == "Hann":
        window_name = "hann"
    elif window_name == "Bohman":
        window_name = "bohman"
    elif window_name == "4-term Blackman-Harris":
        window_name = "blackmanharris"

    spectra_processing = get_fft

    # Get nperseg
    if time_resolution == "Auto":
        nperseg = get_nperseg(samples=samples)
        print(f"Number of points per segment: {nperseg}")
    else:
        nperseg = int(float(time_resolution) * fs)
        print(f"Number of points per segment: {nperseg}")

    # Get noverlap
    noverlap = get_noverlap(overlap=overlap, nperseg=nperseg, window_name=window_name)

    Sxx = get_spectrogram(
        sig, nperseg=nperseg, noverlap=noverlap, window_name=window_name
    )

    SK = get_spectral_kurtosis(Sxx)
