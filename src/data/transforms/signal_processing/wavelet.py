"""
Código base para transformada wavelet contínua e discreta.
"""

import math

import matplotlib.pyplot as plt
import numpy as np
import pywt
import scaleogram
import scipy.fftpack


def wavelet_continuous(sig, fs, plot=False, ylog=False):
    """
    Cálculo da Transformada Wavelet Contínua de um sinal.

    Parameters
    ----------
    sig : np.array
        Sinal de entrada.
    fs : int
        Frequência de amostragem do sinal.
    plot : bool, optional
        Se plota ou não o sinal, by default False
    ylog : bool, optional
        Se coloca a escala do eixo y em log, by default False

    Returns
    -------
    t: np.array
        Eixo do tempo.
    frequencies: np.array
        Eixo da frequência.
    power: np.array
        Potência do sinal.
    """
    y = sig - np.mean(sig)
    N = len(y)
    # amp = np.max(np.abs(y))
    dt = 1 / fs

    freqmin = 10  # Min frequency in Wavelet Spectogram
    freqmax = fs / 2  # Max frequency in Wavelet Spectogram
    # freqview = 100  # Max frequency in "Envelope" graph
    n = 1024  # Max Division of Signal's Amplitude in Wavelet Spectogram

    t = np.linspace(0, N / fs, num=N)
    scales = np.arange(int(fs / freqmax), int(fs / freqmin))
    # %time coefficients, frequencies = scaleogram.fastcwt(y,scales,'cmor1.5-1.0',dt)
    # %time coefficients, frequencies = pywt.cwt(y, scales, 'cmor1.5-1.0', dt)
    coefficients, frequencies = scaleogram.fastcwt(y, scales, "cmor1.5-1.0", dt)
    ind = np.unravel_index(
        np.argmax(np.abs(coefficients), axis=None), coefficients.shape
    )
    power = (abs(coefficients)) ** 2
    # contourlevels = [
    #     amp / n,
    #     amp / (n / 2),
    #     amp / (n / 4),
    #     amp / (n / 8),
    #     amp / (n / 16),
    #     amp / (n / 32),
    #     amp / (n / 64),
    # ]

    # "Envelope" of the Coefficients
    magcoeff = np.abs(coefficients)
    MAG = np.zeros((N, 1), complex)
    MAG[:, 0] = scipy.fftpack.fft(magcoeff[ind[0]][:])
    # MAG_media = 2.0 / N * np.mean(np.abs(MAG[: N // 2, :1]), 1)
    # freq = np.linspace(0.0, int(1.0 / (2.0 * dt)), int(N / 2))

    if plot:
        plt.figure(figsize=(12, 8), facecolor="w")
        plt.contourf(t, frequencies, power, extend="both", cmap=plt.cm.seismic)
        plt.colorbar()
        plt.title(
            f"Wavelet CWT - Morlet, scales= {fs/freqmax}-{fs/freqmin}, divisions: {n}"
        )
        plt.ylabel("Frequency")
        plt.xlabel("Time")
        if ylog:
            plt.yscale("log")

    return t, frequencies, power


def wavelet_discrete(sig, fs, plot=True):
    """
    Cálculo da Transformada Wavelet Discreta de um sinal.

    Parameters
    ----------
    sig : np.array
        Sinal de entrada.
    fs : int
        Frequência de amostragem do sinal.
    plot : bool, optional
        Se plota ou não a transformada, by default True

    Returns
    -------
    time: np.array
        Eixo do tempo.
    freq: np.array
        Eixo da frequência.
    power: np.array
        Potência do sinal.
    """
    y = sig - np.mean(sig)
    N = len(y)
    amp = np.max(np.abs(y))
    # dt = 1 / fs

    freqmin = 50  # Min frequency in Wavelet Spectrogram
    freqmax = fs / 2  # Max frequency in Wavelet Spectrogram
    # freqview = 100  # Max frequency in "Envelope" graph
    n = 1024  # Max Division of Signal's Amplitude in Wavelet Spectogram

    w = pywt.Wavelet("db1")
    lvls = math.ceil(math.log2(N))
    coeffs = pywt.wavedec(y, w, level=lvls)
    cc = np.abs(np.array([coeffs[-1]]))

    for i in range(lvls - 1):
        cc = np.concatenate(
            np.abs([cc, np.array([np.repeat(coeffs[lvls - 1 - i], pow(2, i + 1))])]),
            axis=0,
        )
    ind = np.unravel_index(np.argmax(np.abs(cc), axis=None), cc.shape)
    magcoeff = np.abs(cc)
    time = np.linspace(start=0, stop=N / fs, num=N // 2)
    freq = np.linspace(start=freqmin, stop=freqmax, num=lvls)
    power = (abs(cc)) ** 2

    contourlevels = [
        amp / n,
        amp / (n / 2),
        amp / (n / 4),
        amp / (n / 8),
        amp / (n / 16),
        amp / (n / 32),
        amp / (n / 64),
    ]

    # "Envelope" of the Coefficients
    MAG = np.zeros((N // 2, 1), complex)
    MAG[:, 0] = scipy.fftpack.fft(magcoeff[ind[0]][:])
    # MAG_media = 2.0 / N * np.mean(np.abs(MAG[: N // 2, :1]), 1)
    # freq_ = np.linspace(0.0, int(1.0 / (2.0 * dt)), int(N / 2))

    if plot:
        plt.figure(figsize=(12, 8), facecolor="w")
        plt.contourf(
            time,
            freq,
            np.array(power),
            contourlevels,
            extend="both",
            cmap=plt.cm.seismic,
        )
        plt.colorbar()
        plt.title("Wavelet DWT")
        plt.ylabel("Frequency")
        plt.xlabel("Time")
        plt.yscale("log")

    return time, freq, power


# wavelet_continuous(np.array(acceleration_raw['z']), fs);
# wavelet_discrete(np.array(acceleration_raw['z']), fs);
