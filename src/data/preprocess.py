"""Preprocess functions to apply on Pytorch VibrationDataset"""

import pandas as pd
import numpy as np
import soxr


def subtract_mean(signal):
    """Subtract the mean from the signal."""

    if isinstance(signal, pd.Series):
        return signal - signal.mean()
    elif isinstance(signal, list):
        return [x - sum(signal) / len(signal) for x in signal]
    elif isinstance(signal, np.ndarray):
        return signal - signal.mean()
    else:
        raise TypeError("Unsupported signal type. Must be a pandas Series or list.")


def convert_to_g(signal):
    """Convert signal from m/s^2 to g (gravitational acceleration) units."""

    if isinstance(signal, pd.Series):
        return signal / 9.81
    elif isinstance(signal, list):
        return [x / 9.81 for x in signal]
    elif isinstance(signal, np.ndarray):
        return signal / 9.81
    else:
        raise TypeError("Unsupported signal type. Must be a pandas Series or list.")


def preprocess_cwru(df: pd.DataFrame, resample: bool) -> pd.DataFrame:
    """Resample CWRU signals from 48 kHz to 12 kHz."""

    if df.empty:
        return df
    else:
        processed_df = df.copy()
        processed_df["signal"] = processed_df["signal"].apply(
            lambda x: subtract_mean(x)
        )

        processed_df["fs"] = processed_df["fs"].astype(int)

        # Resample 48 kHz to 12 kHz
        if resample and (
            len(processed_df) > 0
            and processed_df[processed_df["fs"] != 12000].shape[0] > 0
        ):
            processed_df.loc[processed_df["fs"] == 48000, "signal"] = processed_df.loc[
                processed_df["fs"] == 48000, "signal"
            ].apply(lambda x: soxr.resample(x, in_rate=48000, out_rate=12000))
            processed_df["fs"] = 12000

        return processed_df


def preprocess_lva(df: pd.DataFrame, resample: bool) -> pd.DataFrame:
    """Resample LVA signals from 25.6 kHz to 12 kHz."""

    if df.empty:
        return df
    else:
        processed_df = df.copy()
        processed_df["sampling_rate"] = processed_df["sampling_rate"].astype(int)
        processed_df["Signal"] = processed_df["Signal"].apply(
            lambda x: subtract_mean(x)
        )

        # Resample 48 kHz to 12 kHz
        if resample and (
            len(processed_df) > 0
            and processed_df[processed_df["sampling_rate"] != 12000].shape[0] > 0
        ):
            processed_df.loc[processed_df["sampling_rate"] == 25600, "Signal"] = (
                processed_df.loc[
                    processed_df["sampling_rate"] == 25600, "Signal"
                ].apply(lambda x: soxr.resample(x, in_rate=25600, out_rate=12000))
            )
            processed_df["sampling_rate"] = 12000

        return processed_df


def preprocess_paderborn(df: pd.DataFrame, resample: bool = False, crop_pct = 0, stage="train") -> pd.DataFrame:
    """Resample Paderborn signals from 64 kHz to 12 kHz."""

    if df.empty:
        return df
    else:
        processed_df = df.copy()

        # Garantir tipo inteiro
        processed_df["fs"] = processed_df["fs"].astype(int)
        processed_df["vibration"] = processed_df["vibration"].apply(
            lambda x: subtract_mean(x)
        )

        # Resample 64 kHz to 12 kHz
        if resample and processed_df[processed_df["fs"] != 12000].shape[0] > 0:
            processed_df.loc[processed_df["fs"] == 64000, "vibration"] = (
                processed_df.loc[
                    processed_df["fs"] == 64000, "vibration"
                ].apply(lambda x: soxr.resample(x, in_rate=64000, out_rate=12000))
            )
            processed_df["fs"] = 12000

        if (crop_pct > 0) and (stage == "train"):
            def crop_signal(signal, crop_pct):
                n = len(signal)
                print(f"Original signal length: {n}")
                crop_len = int(n * crop_pct)
                crop = signal[0:crop_len]
                print(f"Cropped signal length: {len(crop)}")
                return crop

            processed_df["vibration"] = processed_df["vibration"].apply(
                lambda x: crop_signal(x, crop_pct)
            )

        return processed_df


def preprocess_ottawa(df: pd.DataFrame, resample: bool, crop_pct = 0, stage=None, subtract_mean=False) -> pd.DataFrame:
    """Resample Ottawa signals from 42 kHz to 12 kHz."""

    if df.empty:
        return df
    else:
        processed_df = df.copy()

        processed_df["fs"] = processed_df["fs"].astype(int)

        if subtract_mean:
            processed_df["vibration"] = processed_df["vibration"].apply(
                lambda x: convert_to_g(subtract_mean(x))
            )
            processed_df["vibration"] = processed_df["vibration"].apply(
                lambda x: subtract_mean(x)
            )

        # Resample 48 kHz to 12 kHz
        if resample and (
            len(processed_df) > 0
            and processed_df[processed_df["fs"] != 12000].shape[0] > 0
        ):
            processed_df.loc[processed_df["fs"] == 42000, "vibration"] = (
                processed_df.loc[
                    processed_df["fs"] == 42000, "vibration"
                ].apply(lambda x: soxr.resample(x, in_rate=42000, out_rate=12000))
            )
            processed_df["fs"] = 12000

        
        
        if (crop_pct > 0) and (stage == "train"):
            def crop_signal(signal, crop_pct):
                n = len(signal)
                print(f"Original signal length: {n}")
                crop_len = int(n * crop_pct)
                crop = signal[0:crop_len]
                print(f"Cropped signal length: {len(crop)}")
                return crop

            processed_df["vibration"] = processed_df["vibration"].apply(
                lambda x: crop_signal(x, crop_pct)
            )

            #test_signal_length = processed_df["vibration"].apply(lambda x: len(x))
            #print(f"After cropping, unique signal lengths: {test_signal_length}")
            
        return processed_df


def preprocess_hust(df: pd.DataFrame, resample: bool) -> pd.DataFrame:
    """Resample Hust signals from 51.2 kHz to 12 kHz."""

    if df.empty:
        return df
    else:
        processed_df = df.copy()
        processed_df["data"] = processed_df["data"].apply(
            lambda x: subtract_mean(convert_to_g(x))
        )

        if resample:
            processed_df["data"] = processed_df["data"].apply(
                lambda x: soxr.resample(x, in_rate=51200, out_rate=12000)
            )

        return processed_df


def segment_dataset(
    df: pd.DataFrame, signal_column: str, segment_size: int = 12000, overlap_pct = 0
) -> pd.DataFrame:
    """Segment the dataset into smaller segments of a given length."""

    new_rows_list = []

    # Iterate over each row of the original dataframe
    for _, row in df.iterrows():
        # Extract the long signal array from the row
        original_signal = row[signal_column]

        # Get all other data from the row, excluding the vibration signal itself
        other_data = row.drop(signal_column)


        overlap = int(overlap_pct * segment_size)
        stride = segment_size - overlap
        #num_segments = len(original_signal) // segment_size
        num_segments = 1 + (len(original_signal) - segment_size) // (
                    segment_size - overlap
                )

        num_segments = max(1, num_segments)

        # Create a new row for each full segment
        for i in range(num_segments):
            

            new_segment = original_signal[
                    i * stride : i * stride + segment_size
                ]

            # Create a new row by combining the other data with the new segment
            new_row = other_data.to_dict()
            new_row[signal_column] = new_segment

            # Add the newly created row to our list
            new_rows_list.append(new_row)

    # Create the final, expanded dataframe from our list of new rows
    expanded_df = pd.DataFrame(new_rows_list)
    expanded_df['waveform_id_seg'] = expanded_df['waveform_id'] + '_' + (expanded_df.index + 1).astype(str)

    return expanded_df.reset_index(drop=True)
