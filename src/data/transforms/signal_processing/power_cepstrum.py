"""
Script to compute the power cepstrum of a time-domain signal.
"""

import numpy as np
from scipy.fft import fft, fftfreq
from scipy.signal import detrend
import torch


def get_fft(
    data, n=None, axis=0, return_onesided=True, remove_mean=True, reduce_size=False
):
    """
    Compute the one-dimensional discrete Fourier transform

    Parameters
    ----------
    data : array_like
        Input data.
    n : int, opt
        Number of samples.
    axis : int, opt
        Axis along which the FFT is computed.
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
    sp = fft(data, n=n, axis=axis) / n

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


def get_fft_torch(
    data,
    n=None,
    axis=0,
    return_onesided=True,
    remove_mean=True,
    reduce_size=False,
):
    """
    Compute the one-dimensional discrete Fourier transform using PyTorch on GPU.

    Parameters
    ----------
    data : torch.Tensor
        Input data (must be a 1D or 2D torch tensor).
    n : int, optional
        Number of FFT points (zero-padding or truncation).
    axis : int, optional
        Axis along which the FFT is computed (default is 0).
    return_onesided : bool, optional
        If True, returns a one-sided spectrum (valid for real-valued input).
    remove_mean : bool, optional
        If True, removes the average signal before computing FFT.
    reduce_size : bool, optional
        If True, rounds FFT results to a given number of decimals.

    Returns
    -------
    torch.Tensor
        Tensor of amplitudes of the discrete Fourier transform coefficients.
    """

    if not isinstance(data, torch.Tensor):
        raise ValueError("Input data should be a torch tensor")

    # Set number of samples
    original_len = data.shape[axis]
    if n is None:
        n = original_len
    elif n < original_len:
        raise ValueError(
            "Number of samples (n) should be greater than or equal to the original vector length"
        )

    # Remove mean if required
    if remove_mean:
        data = data - data.mean(dim=axis, keepdim=True)

    # Compute FFT (normalized by `n`)
    sp = torch.fft.fft(data, n=n, dim=axis) / n

    # Correction factor in case of zero-padding
    sp = sp * (n / original_len)

    # Compute one-sided FFT if requested
    if return_onesided:
        N_half = n // 2
        out = torch.abs(sp[:N_half])  # Extract positive frequencies
        out[1:] *= (
            2  # Multiply by 2 to preserve energy conservation (except DC component)
        )
    else:
        out = torch.abs(sp)

    # Reduce precision if needed
    if reduce_size:
        out = torch.round(out * 1e5) / 1e5  # Round to 5 decimals

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


def get_cepstrum(data):
    """
    Return the cepstrum (spectrum of a logarithmic power spectrum).

    Parameters
    ----------
    data : array_like
        Time-domain signal
    fs : int, opt
        Sampling rate

    Returns
    -------
    np.ndarray
        Array of cepstral amplitudes
    """

    # Compute the spectrum
    X = get_fft_torch(data, remove_mean=False)
    # X = get_fft(data, remove_mean=False)

    # Compute frequency vector
    # freq_vector = get_fftfreq(data.size, fs)

    # Return the logarithm of the power spectrum
    log_X = get_logarithm_spectrum_torch(X**2)
    # log_X = get_logarithm_spectrum(X**2)

    # Return the spectrum of the logarithmic power spectrum
    cepstrum = get_fft_torch(log_X)

    return cepstrum  # , freq_vector


def get_logarithm_spectrum(X, dB_scale=True, remove_mean=True, remove_trend=True):
    """
    Compute the logarithm of the spectrum

    Parameters
    ----------
    X : ndarray
        Array of spectral amplitudes
    dB_scale : bool, opt
        If True, applies the formula 10*log10(X).
    remove_mean : bool, opt
        If True, it removes the mean value of the logarithmic spectrum
    remove_trend : bool, opt
        If True, it removes the linear trend of the logarithmic spectrum

    Returns
    -------
    np.ndarray
        Array of logarithmic spectral amplitudes
    """

    # Substitute zero values in the FFT spectrum to avoid infinite
    # values in the log operation
    MIN_VALUE = 1e-16
    X = np.where(X == 0, MIN_VALUE, X)

    # Compute the logarithm (base 10) of the spectral amplitudes
    if dB_scale:
        log_X = 10 * np.log10(X)
    else:
        log_X = np.log10(X)

    # Remove the mean value of the logarithmic spectrum
    if remove_mean:
        log_X = log_X - np.mean(log_X)

    # Remove the linear trend of the logarithmic spectrum
    if remove_trend:
        log_X = detrend(log_X)

    return log_X


def detrend_torch(x: torch.Tensor) -> torch.Tensor:
    """
    Remove linear trend from 1D or 2D torch tensor.
    If x is 2D, it assumes shape [B, N], detrending along dim=-1.
    """
    if x.dim() == 1:
        x = x.unsqueeze(0)

    B, N = x.shape
    t = torch.linspace(0, 1, N, device=x.device, dtype=x.dtype).expand(B, -1)
    ones = torch.ones_like(t)
    A = torch.stack([t, ones], dim=2)  # [B, N, 2]

    x_unsq = x.unsqueeze(2)  # [B, N, 1]

    # Solve least squares: (A^T A)^-1 A^T x
    At = A.transpose(1, 2)  # [B, 2, N]
    AtA = At @ A  # [B, 2, 2]
    Atx = At @ x_unsq  # [B, 2, 1]
    coeffs = torch.linalg.solve(AtA, Atx)  # [B, 2, 1]
    trend = torch.sum(A * coeffs.transpose(1, 2), dim=2)  # [B, N]

    detrended = x - trend
    return detrended.squeeze(0) if detrended.shape[0] == 1 else detrended


def get_logarithm_spectrum_torch(
    X: torch.Tensor,
    dB_scale: bool = True,
    remove_mean: bool = True,
    remove_trend: bool = True,
) -> torch.Tensor:
    """
    Compute the logarithm of the spectrum in PyTorch.

    Parameters
    ----------
    X : torch.Tensor
        Spectral amplitudes (1D or 2D tensor)
    dB_scale : bool
        If True, applies 10*log10(X), else log10(X)
    remove_mean : bool
        Whether to subtract the mean of the log spectrum
    remove_trend : bool
        Whether to remove a linear trend (detrend)

    Returns
    -------
    torch.Tensor
        Logarithmic spectrum
    """

    # Avoid log(0)
    MIN_VALUE = 1e-16
    X = torch.clamp(X, min=MIN_VALUE)

    # Logarithmic scale
    if dB_scale:
        log_X = 10.0 * torch.log10(X)
    else:
        log_X = torch.log10(X)

    # Remove mean
    if remove_mean:
        log_X = log_X - log_X.mean(dim=-1, keepdim=True)

    # Remove trend
    if remove_trend:
        log_X = detrend_torch(log_X)

    return log_X


def quefrequencies(samples, fs):
    """
    Return the Quefrency vector

    Parameters
    ----------
    samples : int
        Number of samples.
    fs : int
        Sampling rate.

    Returns
    -------
    np.ndarray
        Quefrency vector
    """
    return get_fftfreq(samples // 2, (samples / 2) / (fs / 2))
