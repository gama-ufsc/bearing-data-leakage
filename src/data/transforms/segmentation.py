import torch
import torch.nn.functional as F


def segment_signals(
    signal: torch.Tensor, segment_size: int, overlap_ratio: float = 0.5
) -> torch.Tensor:
    """
    Segments a batched 1D signal into chunks, padding with zeros if necessary.

    Args:
        signal (torch.Tensor): Input tensor of shape (batch_size, 1, signal_len)
        segment_size (int): Size of each segment (number of samples).
        overlap_ratio (float): Fraction of overlap (0.0 means no overlap, >0.0 applies overlap).

    Returns:
        torch.Tensor: Segmented tensor of shape (batch_size, 1, n_segments, segment_size)
    """
    batch_size, channels, signal_len = signal.shape
    assert channels == 1, "Input tensor must have shape (batch_size, 1, signal_len)"
    assert 0.0 <= overlap_ratio < 1.0, "overlap_ratio must be in [0.0, 1.0)"

    if overlap_ratio == 0.0:
        # === No Overlap ===
        n_segments = (signal_len + segment_size - 1) // segment_size  # ceil division
        total_len = n_segments * segment_size
        pad_len = total_len - signal_len

        if pad_len > 0:
            signal = F.pad(signal, (0, pad_len))  # (left, right) padding

        output = signal.view(batch_size, 1, n_segments, segment_size)

    else:
        # === With Overlap ===
        step = int(segment_size * (1 - overlap_ratio))
        if step <= 0:
            raise ValueError("Overlap ratio too high, must be less than 1.0")

        last_start = (signal_len - segment_size) // step * step
        needed_len = last_start + segment_size
        pad_len = max(0, needed_len - signal_len)

        if pad_len > 0:
            signal = F.pad(signal, (0, pad_len))

        # Recalculate signal_len after padding
        signal_len = signal.shape[-1]

        segments = []
        for start_idx in range(0, signal_len - segment_size + 1, step):
            segment = signal[:, :, start_idx : start_idx + segment_size]
            segments.append(segment.unsqueeze(2))  # add segment dimension at position 2

        output = torch.cat(segments, dim=2)

    return output
