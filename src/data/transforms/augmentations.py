import torch.nn as nn
import torch
import torchaudio.transforms as T
import torch.nn.functional as Fnn
# import random


class RandomCrop1D(nn.Module):
    def __init__(self, crop_size):
        super(RandomCrop1D, self).__init__()
        self.crop_size = crop_size

    def __call__(self, signal):
        if isinstance(signal, torch.Tensor):
            length = signal.shape[-1]
        else:
            raise TypeError("Input should be a torch.Tensor")

        if length < self.crop_size:
            raise ValueError("Crop size should be smaller than the signal length")

        start = torch.randint(low=0, high=length - self.crop_size, size=(1, 1))
        return signal[..., start : start + self.crop_size]  # Works for (C, L) or (L)v


class VariableRandomCrop1D(nn.Module):
    def __init__(self, eps, standard_crop, chance=0.5):
        super(VariableRandomCrop1D, self).__init__()
        self.eps = eps
        self.standard_crop = standard_crop
        self.chance = chance
        self.min_crop_size = int((1 - self.eps) * standard_crop)
        self.max_crop_size = int((1 + self.eps) * standard_crop)

    def __call__(self, signal):
        if isinstance(signal, torch.Tensor):
            length = signal.shape[-1]
        else:
            raise TypeError(
                f"Input should be a torch.Tensor, but is of type {type(signal)}"
            )

        if length <= self.max_crop_size:
            print(length)
            raise ValueError(
                "Max crop size should be equal or smaller than the signal length"
            )

        # Apply a self.chance of using the variable crop size
        if torch.rand(1) > self.chance:
            crop_size = torch.randint(
                low=self.min_crop_size, high=self.max_crop_size, size=(1, 1)
            )
        else:
            crop_size = self.standard_crop

        start = torch.randint(low=0, high=length - crop_size, size=(1, 1))

        return signal[..., start : start + crop_size]  # Works for (C, L) or (L)

        # crop_size = random.randint(self.min_crop_size, self.max_crop_size)
        # start = random.randint(0, length - crop_size)


class CenterCrop1D(nn.Module):
    def __init__(self, crop_size):
        super(CenterCrop1D, self).__init__()
        self.crop_size = crop_size

    def __call__(self, signal):
        """
        Crops the center of a 1D signal.

        Args:
            signal (torch.Tensor): Input signal of shape (C, L) or (L,).

        Returns:
            torch.Tensor: Cropped signal of shape (C, crop_size) or (N, C, crop_size).
        """
        if not isinstance(signal, torch.Tensor):
            raise TypeError("Input should be a torch.Tensor")

        length = signal.shape[-1]  # Assuming (C, L) or (L,)

        if length < self.crop_size:
            raise ValueError("Crop size should be smaller than the signal length")

        start = (length - self.crop_size) // 2  # Center crop start index
        return signal[..., start : start + self.crop_size]  # Works for (C, L) or (L)


class CustomTimeStretchTorchAudio(nn.Module):
    def __init__(self, desired_length: int = 8192):
        super(CustomTimeStretchTorchAudio, self).__init__()
        self.desired_length = desired_length

    def __call__(self, signal):
        """
        Applies time stretching to a 1D signal using TimeStretch, Spectrogram and InverseSpectrogram.

        Args:
            signal (torch.Tensor): Input signal of shape (C, L) or (L,).

        Returns:
            torch.Tensor: Time-stretched signal of shape (C, L) or (L,).
        """
        if not isinstance(signal, torch.Tensor):
            raise TypeError("Input should be a torch.Tensor")

        # if signal.ndim == 1:
        #    signal = signal.unsqueeze(0)  # Add channel dimension

        # if signal.ndim != 2:
        #    raise ValueError("Input signal should have shape (C, L) or (L,)")

        ratio = signal.shape[-1] / self.desired_length

        spectrogram = T.Spectrogram(power=None)
        stretch = T.TimeStretch()
        stft = spectrogram(signal)
        stft_stretched = stretch(stft, ratio)

        inverse = T.InverseSpectrogram()
        reconstructed = inverse(stft_stretched)

        if reconstructed.shape[0] > self.desired_length:
            return reconstructed[..., : self.desired_length]
        elif reconstructed.shape[0] < self.desired_length:
            return torch.cat(
                [
                    reconstructed,
                    torch.zeros(
                        signal.shape[0], self.desired_length - reconstructed.shape[0]
                    ),
                ],
                dim=-1,
            )
        else:
            return reconstructed


class CustomResample(nn.Module):
    def __init__(self, desired_shape: int = 8192, signal_fs: float = 2048):
        super(CustomResample, self).__init__()

        self.signal_fs = signal_fs
        self.desired_shape = desired_shape

    def __call__(self, signal):
        """
        Resamples a 1D signal using Resample.

        Args:
            signal (torch.Tensor): Input signal of shape (C, L) or (L,).

        Returns:
            torch.Tensor: Resampled signal of shape (C, L) or (L,).
        """
        if not isinstance(signal, torch.Tensor):
            raise TypeError("Input should be a torch.Tensor")

        # signal_len = signal.shape[-1]
        # duration = signal_len / self.signal_fs

        # new_freq = np.ceil(self.desired_shape/duration)
        # resampled_sig = F.resample(waveform=signal, orig_freq=self.signal_fs, new_freq=new_freq)
        if len(signal.shape) == 1:
            signal = signal.unsqueeze(0).unsqueeze(0)

        resampled_sig = Fnn.interpolate(
            signal, size=self.desired_shape, mode="linear", align_corners=False
        ).flatten()

        if resampled_sig.shape[-1] > self.desired_shape:
            resampled_sig = resampled_sig[..., : self.desired_shape]
        elif resampled_sig.shape[-1] < self.desired_shape:
            print(f"Padding {self.desired_shape - resampled_sig.shape[-1]}")
            resampled_sig = torch.nn.functional.pad(
                resampled_sig, (0, self.desired_shape - resampled_sig.shape[-1])
            )
        return resampled_sig


class Padding(nn.Module):
    def __init__(self, desired_shape: int = 8192):
        super(Padding, self).__init__()
        self.desired_shape = desired_shape

    def __call__(self, signal):
        """
        Pads a 1D signal to the desired shape.

        Args:
            signal (torch.Tensor): Input signal of shape (C, L) or (L,).

        Returns:
            torch.Tensor: Padded signal of shape (C, L) or (L,).
        """
        if not isinstance(signal, torch.Tensor):
            raise TypeError("Input should be a torch.Tensor")

        if signal.shape[-1] > self.desired_shape:
            return signal[..., : self.desired_shape]
        elif signal.shape[-1] < self.desired_shape:
            return torch.nn.functional.pad(
                signal, (0, self.desired_shape - signal.shape[-1])
            )
        else:
            return signal


class Padding2(nn.Module):
    def __init__(self, min_crop_size, max_crop_size):
        super(Padding2, self).__init__()
        self.min_crop_size = min_crop_size
        self.max_crop_size = max_crop_size

    def __call__(self, signal):
        if not isinstance(signal, torch.Tensor):
            raise TypeError(
                f"Input should be a torch.Tensor, but is of type {type(signal)}"
            )

        length = signal.shape[-1]

        if length < self.max_crop_size:
            raise ValueError("Max crop size should be smaller than the signal length")

        crop_size = torch.randint(
            low=self.min_crop_size, high=self.max_crop_size, size=(1, 1)
        )
        start = torch.randint(low=0, high=length - crop_size, size=(1, 1))

        # Create a mask of the same shape as the signal
        mask = torch.zeros_like(signal)

        # Preserve only the randomly selected window
        mask[..., start : start + crop_size] = 1

        # Apply mask to the signal
        masked_signal = signal * mask

        return masked_signal


class Padding3(nn.Module):
    def __init__(self, min_crop_size, max_crop_size, continuous=True):
        super(Padding3, self).__init__()
        self.min_crop_size = min_crop_size
        self.max_crop_size = max_crop_size
        self.continuous = continuous

    def __call__(self, signal):
        if not isinstance(signal, torch.Tensor):
            raise TypeError(
                f"Input should be a torch.Tensor, but is of type {type(signal)}"
            )

        length = signal.shape[-1]

        if length < self.max_crop_size:
            raise ValueError("Max crop size should be smaller than the signal length")

        crop_size = torch.randint(
            low=self.min_crop_size, high=self.max_crop_size, size=(1, 1)
        )
        start = torch.randint(low=0, high=length - crop_size, size=(1, 1))

        # Create a mask of the same shape as the signal
        mask = torch.zeros_like(signal)

        # Preserve only the randomly selected window
        mask[..., start : start + crop_size] = 1

        # Apply mask to the signal
        masked_signal = signal * mask

        # Limit the shift to avoid discontinuities
        if self.continuous is True:
            max_shift = length - crop_size
            shift = torch.randint(
                low=0, high=max_shift, size=(1, 1)
            )  # Ensure the shift does not wrap around
            # Perform the shift
            shifted_signal = torch.zeros_like(signal)
            shifted_signal[..., shift : shift + crop_size] = masked_signal[
                ..., start : start + crop_size
            ]
        else:
            # Randomly shift the masked signal
            shift = torch.randint(
                low=0, high=length - 1, size=(1, 1)
            )  # Random shift within range
            shifted_signal = torch.roll(masked_signal, shifts=shift, dims=-1)

        return shifted_signal


class RandomGainOnFFT(nn.Module):
    def __init__(self, mean=1, std=0.3):
        super(RandomGainOnFFT, self).__init__()
        self.mean = mean
        self.std = std

    def __call__(self, fft):
        """
        Applies a single constant random gain to the magnitude of the FFT.

        Args:
            fft (torch.Tensor): Input FFT of shape (C, F).

        Returns:
            torch.Tensor: FFT with random gain applied.
        """
        if not isinstance(fft, torch.Tensor):
            raise TypeError("Input should be a torch.Tensor")

        gain = torch.normal(
            mean=self.mean, std=self.std, size=(1, 1)
        ).squeeze()  # .to(fft.device)

        return fft * gain


class RandomGainOnFFTBins(nn.Module):
    def __init__(self, mean=1, std=0.3):
        super(RandomGainOnFFTBins, self).__init__()
        self.mean = mean
        self.std = std

    def __call__(self, fft):
        """
        Applies a random gain to the magnitude of the FFT.

        Args:
            fft (torch.Tensor): Input FFT of shape (C, F).

        Returns:
            torch.Tensor: FFT with random gain applied.
        """
        if not isinstance(fft, torch.Tensor):
            raise TypeError("Input should be a torch.Tensor")

        gain = (
            torch.normal(mean=self.mean, std=self.std, size=(1, fft.shape[-1]))
            .to(fft.device)
            .squeeze()
        )

        return fft * gain


class RandomPitchShift(nn.Module):
    def __init__(self, max_n_steps=15, sample_rate=2048):
        super(RandomPitchShift, self).__init__()
        self.max_n_steps = max_n_steps
        self.sample_rate = sample_rate

    def __call__(self, signal):
        """
        Applies pitch shifting to a 1D signal using TimeStretch.

        Args:
            signal (torch.Tensor): Input signal of shape (C, L) or (L,).

        Returns:
            torch.Tensor: Pitch-shifted signal of shape (C, L) or (L,).
        """
        if not isinstance(signal, torch.Tensor):
            raise TypeError("Input should be a torch.Tensor")

        n_steps = torch.randint(low=0, high=self.max_n_steps, size=(1, 1))

        torch.use_deterministic_algorithms(True, warn_only=True)
        with torch.no_grad():
            shifted_signal = T.PitchShift(
                sample_rate=self.sample_rate, n_steps=n_steps
            ).to(signal.device)(signal)

        return shifted_signal


class RandomExclusiveAugmentation:
    def __init__(self, aug_sequences, probs):
        """
        Randomly applies one of the augmentation sequences or none using PyTorch functions.
        :param aug_sequences: List of augmentation pipelines (torchvision.transforms.Compose objects)
        :param probs: Probability of selecting each augmentation sequence
        """
        self.aug_sequences = (
            aug_sequences  # List of transforms (can be individual or composed)
        )
        self.probs = torch.tensor(probs)  # Torch tensor of probabilities

    def __call__(self, img):
        idx = torch.multinomial(self.probs, num_samples=1, replacement=True).item()
        return self.aug_sequences[idx](img)


class RandomFlip(nn.Module):
    def __init__(self, p=0.5):
        super(RandomFlip, self).__init__()
        self.p = p

    def __call__(self, signal):
        """
        Randomly flips a 1D signal.

        Args:
            signal (torch.Tensor): Input signal of shape (C, L) or (L,).

        Returns:
            torch.Tensor: Flipped signal of shape (C, L) or (L,).
        """
        if not isinstance(signal, torch.Tensor):
            raise TypeError("Input should be a torch.Tensor")

        if torch.rand(1) < self.p:
            return -signal
        else:
            return signal
