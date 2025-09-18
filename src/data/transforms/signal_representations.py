"""
1D and 2D signal representation functions to be used with our experiments.

Such as:
    - Time
    - Spectrum
    - Cepstrum
    - Spectrogram
    - Continuous Wavelet
    - Cepstrogram
    - Modulation Spectrogram
"""

import numpy as np
import torch

# Check https://github.com/MuSAELab/amplitude-modulation-analysis-module for more details
# from am_analysis import am_analysis
from scipy import signal

from .signal_processing.power_cepstrum import (
    get_cepstrum,
    get_fft,
    get_fft_torch,
    get_logarithm_spectrum,
)
from .signal_processing.spectrogram_and_sk import (
    get_noverlap,
    get_nperseg,
    get_spectral_kurtosis,
    get_spectrogram,
)

from .kurtogram import kurtogram_bandpass

BASE_SPECTROGRAM_PARAMS = {
    "fs": 12000,
    "window": signal.windows.hann(104),
    "overlap": 54,
    "nfft": 452,
    "crop": True,
    "mode": "magnitude",
    "scaling": "spectrum",
}

BASE_SPECTRAL_KURTOSIS_PARAMS = {
    "samples": 4096,
    "fs": 12000,
    "window_name": "hann",
    "overlap": "Auto",
    "remove_zero_freq": False,
    "remove_Nyquist_freq": True,
}

BASE_CEPSTROGRAM_PARAMS = {
    "fs": 12000,
    "nperseg": 179,
    "noverlap": 89,
    "window_name": "hann",
    "remove_zero_freq": False,
    "remove_Nyquist_freq": True,
    "remove_mean": True,
    "remove_trend": False,
}

BASE_MOD_SPECTROGRAM_PARAMS = {
    "fs": 12000,
    "noverlap": 176,
    "nperseg": 256,
    "transform_name": "nperseg-256-noverlap-176",
}


def time(arr: np.array, **params) -> np.array:
    """
    Apply time transformation to the input array.

    Parameters
    ----------
    arr : np.array
        Input array to be transformed.
    **params : dict
        Parameters for the transformation.

    Returns
    -------
    np.array
        Transformed array.
    """
    return arr


def fft(arr: np.array, **params) -> np.array:
    """
    Apply Fast Fourier Transform (FFT) to the input array.

    Parameters
    ----------
    arr : np.array
        Input array to be transformed.
    **params : dict
        Parameters for the transformation.

    Returns
    -------
    np.array
        Transformed array.
    """
    if isinstance(arr, np.ndarray):
        return get_fft(arr)
    elif isinstance(arr, torch.Tensor):
        return get_fft_torch(arr)


def cepstrum(arr: np.array, **params) -> np.array:
    """
    Apply cepstrum transformation to the input array.

    Parameters
    ----------
    arr : np.array
        Input array to be transformed.
    **params : dict
        Parameters for the transformation.

    Returns
    -------
    np.array
        Transformed array.
    """

    cepstrum = get_cepstrum(arr)
    return cepstrum


def spectrogram(arr: np.array, **params) -> np.array:
    """
    Apply spectrogram transformation to the input array.

    Parameters
    ----------
    arr : np.array
        Input array to be transformed.
    **params : dict
        Parameters for the transformation.

    Returns
    -------
    np.array
        Transformed array.
    """
    f, t, Sxx = signal.spectrogram(
        arr,
        fs=12000,
        window=params["window"],
        nfft=params["nfft"],
        noverlap=params["overlap"],
        mode=params["mode"],
        scaling=params["scaling"],
    )
    if params.get("crop"):
        Sxx = Sxx[:-3, :-4]
    return Sxx


def spectrogram_tensor(
    x: torch.Tensor,
    fs=64000,
    window=signal.windows.hann(104),
    overlap=54,
    nfft=452,
    crop=True,
    mode="magnitude",
    scaling="spectrum",
) -> torch.Tensor:
    """
    Apply spectrogram transformation to a batched tensor.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor of shape [B, C, T].
    **params : dict
        Parameters for the spectrogram computation.

    Returns
    -------
    torch.Tensor
        Spectrogram tensor of shape [B, C, H, W].
    """
    B, C, T = x.shape
    x_np = x.detach().cpu().numpy()  # Convert to numpy for scipy

    specs = []
    for b in range(B):
        channels = []
        for c in range(C):
            f, t_spec, Sxx = signal.spectrogram(
                x_np[b, c],
                fs=fs,
                window=window,
                nfft=nfft,
                noverlap=overlap,
                mode=mode,
                scaling=scaling,
            )
            channels.append(Sxx)
        specs.append(channels)

    specs_np = np.array(specs)  # Shape: [B, C, H, W]
    specs_tensor = torch.tensor(specs_np, dtype=x.dtype, device=x.device)
    return specs_tensor


def spectrogram_with_freq_time(arr: np.array, **params) -> np.array:
    """
    Apply spectrogram transformation to the input array and return the frequency and time axes.

    Parameters
    ----------
    arr : np.array
        Input array to be transformed.
    **params : dict
        Parameters for the transformation.

    Returns
    -------
    np.array
        Transformed array.
    """
    f, t, Sxx = signal.spectrogram(
        arr,
        fs=12000,
        window=params["window"],
        nfft=params["nfft"],
        noverlap=params["overlap"],
        mode=params["mode"],
        scaling=params["scaling"],
    )
    if params.get("crop"):
        Sxx = Sxx[:-3, :-4]
    return Sxx, f, t


def cepstrogram(arr: np.array, **params) -> np.array:
    """
    Apply cepstrogram transformation to the input array.

    Parameters
    ----------
    arr : np.array
        Input array to be transformed.
    **params : dict
        Parameters for the transformation.

    Returns
    -------
    np.array
        Transformed array.
    """
    noverlap = params["nperseg"] - params["stride"]

    Sxx, _, _ = get_spectrogram(
        arr,
        fs=params["fs"],
        nperseg=params["nperseg"],
        noverlap=noverlap,
        window_name=params["window_name"],
        remove_zero_freq=params["remove_zero_freq"],
        remove_Nyquist_freq=params["remove_Nyquist_freq"],
    )
    log_Sxx = get_logarithm_spectrum(
        Sxx**2, remove_mean=params["remove_mean"], remove_trend=params["remove_trend"]
    )
    Cxx = get_fft(log_Sxx, axis=0)
    # quefrency = quefrequencies(nperseg, fs)
    if params["cut_bins"] > 0:
        Cxx = Cxx[params["cut_bins"] :, :]
    return Cxx


def spectral_kurtosis(arr: np.array, **params) -> np.array:
    """
    Calculate the spectral kurtosis of the input array.

    Parameters
    ----------
    arr : np.array
        Input array to be transformed.
    **params : dict
        Parameters for the transformation.

    Returns
    -------
    np.array
        Transformed array.
    """
    nperseg = get_nperseg(samples=params["samples"])
    noverlap = get_noverlap(
        overlap=params["overlap"], nperseg=nperseg, window_name=params["window_name"]
    )
    Sxx, _, _ = get_spectrogram(
        arr,
        fs=params["fs"],
        nperseg=nperseg,
        noverlap=noverlap,
        window_name=params["window_name"],
        remove_zero_freq=params["remove_zero_freq"],
        remove_Nyquist_freq=params["remove_Nyquist_freq"],
    )
    return get_spectral_kurtosis(Sxx)

    # def modulation_spectrogram(arr: np.array, **params):
    """
    Apply modulation spectrogram transformation to the input array.

    Parameters
    ----------
    arr : np.array
        Input array to be transformed.
    **params : dict
        Parameters for the transformation.

    Returns
    -------
    np.array
        Transformed array.
    """
    # win_shift = params["nperseg"] - params["noverlap"]
    # win_shift = params["stride"]
    # mod_spectrogram = am_analysis.strfft_modulation_spectrogram(
    #    arr, params["fs"], params["nperseg"], win_shift
    # )
    # X_pwr = mod_spectrogram["power_modulation_spectrogram"]
    # X_plot = 10 * np.log10(X_pwr[:, :] + np.finfo(float).eps)[:, :, 0]
    # return X_plot


##### ENVELOPE FUNCTIONS #####


def bandpass_filter(
    input_signal, N, cutoff, fs, btype="bandpass", analog=False, output="sos"
):
    """
    This function is designed to apply a butterworth filter in a given input signal,
    where the main input variables are:

    input_signal:array_like
    The signal to be filtered (it must be a array)

    N: int or array_like

    cutoff: array_like or list
    Cutoff frequency(ies) in Hz (int). Note that a scalar or length-2
    sequence giving the critical frequencies.

    fs : float
    The sampling frequency of the digital system.

    btype, analog, output: same parameters describe in scipy.signal.butter.

    check_response: boolean
    Plot the frequency response of a digital filter.

    Returns
    -------

    output_signal: ndarray
    Filtered signal.
    """

    if output == "zpk":
        raise Exception(
            'Zero and poles output is not implemented in this version! \
                        You must use "ba" or "sos" output.'
        )

    nyq = 0.5 * fs
    Wn = np.asarray(cutoff, dtype=np.float64) / nyq

    butter_coef = signal.butter(N, Wn, btype=btype, analog=analog, output=output)

    if output == "sos":
        return signal.sosfiltfilt(
            butter_coef, input_signal, padlen=int(N / 2), padtype="even"
        )

    elif output == "ba":
        return signal.filtfilt(butter_coef[0], butter_coef[1], input_signal)


def array_filter(array, fs, cutoff, filt_order=3):
    """
    Apply bandpass or highpass filter (if necessary) to the array.
    """

    if cutoff == "custom":
        cutoff = [0, fs / 2 * 0.97]

    lowcut = float(cutoff[0])
    highcut = float(cutoff[1])
    nyq = fs * 0.5
    Wn_lowcut = lowcut / nyq
    Wn_highcut = highcut / nyq

    if (0 < Wn_lowcut < 1) and (0 < Wn_highcut < 1):
        filtered_array = bandpass_filter(
            array, filt_order, [lowcut, highcut], fs, btype="bandpass"
        )
    elif (0 < Wn_lowcut < 1) and (Wn_highcut >= 1):
        filtered_array = bandpass_filter(
            array, filt_order, lowcut, fs, btype="highpass"
        )
    elif (Wn_lowcut <= 0) and (0 < Wn_highcut < 1):
        filtered_array = bandpass_filter(
            array, filt_order, highcut, fs, btype="lowpass"
        )
    else:
        filtered_array = np.zeros(array.shape)

    return filtered_array


def get_envelope(
    x: np.array,
    sampling_rate: float,
    domain="frequency",
    cutoff=[100, 500],
    order: int = 12,
    to_tensor: bool = True,
    **kwargs,
):
    """
    Compute the envelope of a signal.
    """

    if isinstance(x, torch.Tensor):
        device = x.device
        x = x.detach().cpu().numpy()

    if cutoff == "kurtogram":
        kurtogram_values = kurtogram_bandpass(x, sampling_rate)
        print(kurtogram_values)

        filtered_signal = array_filter(
            x,
            sampling_rate,
            kurtogram_values,
            filt_order=order,
        )
    elif cutoff is None:
        filtered_signal = x
    else:
        filtered_signal = array_filter(x, sampling_rate, cutoff, filt_order=order)

    temporal_envelope = np.abs(signal.hilbert(filtered_signal))

    if domain == "time":
        envelope = temporal_envelope
    elif domain == "frequency":
        envelope = get_fft(temporal_envelope)
    else:
        raise ValueError(
            "Envelope invalid domain. Choose between 'time' and 'frequency'."
        )

    if isinstance(envelope, np.ndarray) & to_tensor:
        envelope = torch.FloatTensor(envelope).to(device)

    return envelope
