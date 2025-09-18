import torch
import numpy as np
from .signal_processing.power_cepstrum import get_fft_torch
from .signal_representations import spectrogram_tensor
from scipy.signal import windows


class MixupCollateFn:
    def __init__(
        self, alpha: float = 0.4, fft: bool = False, stft: bool = False, fs=64e3
    ):
        self.alpha = alpha
        self.fft = fft
        self.fs = fs
        self.stft = stft

    def __call__(self, batch):
        xs = torch.stack([item["X"] for item in batch])
        ys = torch.stack([item["label"] for item in batch])  # Already one-hot encoded

        lam = np.random.beta(self.alpha, self.alpha)
        index = torch.randperm(xs.size(0))

        mixed_xs = lam * xs + (1 - lam) * xs[index]
        if self.fft:
            mixed_xs = get_fft_torch(mixed_xs, axis=2)
        elif self.stft:
            mixed_xs = spectrogram_tensor(
                mixed_xs,
                fs=self.fs,
                window=windows.hann(104),
                nfft=452,
                overlap=54,
                crop=True,
                mode="magnitude",
                scaling="spectrum",
            )

        mixed_ys = lam * ys + (1 - lam) * ys[index]

        idxs = torch.tensor([item["idx"] for item in batch])

        return {"X": mixed_xs, "label": mixed_ys, "idx": idxs}


class STFTonBatch:
    def __init__(self, fs=64e3):
        self.fs = fs

    def __call__(self, batch):
        xs = torch.stack([item["X"] for item in batch])
        ys = torch.stack([item["label"] for item in batch])  # Already one-hot encoded

        xs = spectrogram_tensor(
            xs,
            fs=self.fs,
            window=windows.hann(104),
            nfft=452,
            overlap=54,
            crop=True,
            mode="magnitude",
            scaling="spectrum",
        )

        idxs = torch.tensor([item["idx"] for item in batch])

        return {"X": xs, "label": ys, "idx": idxs}
