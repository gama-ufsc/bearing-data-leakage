import pandas as pd
import numpy as np


## aux function
def add_groups(df: pd.DataFrame, n=1) -> pd.DataFrame:
    df = df.copy()
    config_cols = ["load_torque", "radial_force", "rotational_speed", "severity"]
    df["config_str"] = df[config_cols].astype(str).agg("_".join, axis=1)

    df["condition_id"] = df.groupby("bearing_id")["config_str"].transform(
        lambda x: pd.Categorical(x).codes
    )

    df.drop(columns=["config_str"], inplace=True)
    return df


####

def extract_1s_segments(signal, signal_pct, end_portion, fs=12000):
            segment_length = fs * 1  # 1 second segment
            if not end_portion:
                return signal[:int(signal_pct*len(signal))]
            else:
                length_to_use = int(signal_pct * len(signal)) // segment_length * segment_length
                if (length_to_use == 0) & (int(signal_pct * len(signal)) < segment_length):
                    print("Warning: signal_pct is too low to extract a full 1-second segment. Adjusting to use at least one segment.")
                    used_samples = int((1-signal_pct) * len(signal))
                    remaining_samples = len(signal) - used_samples
                    print(f"Original signal length: {len(signal)}, Remaining samples after applying signal_pct: {remaining_samples}")
                    # pad remaining samples with zero to guarantee at least one segment of 1s                    
                    if remaining_samples < segment_length:
                        padding = segment_length - remaining_samples
                        signal = np.pad(signal[-remaining_samples:], (padding, 0), mode='constant')
                    else:
                        signal = signal[-segment_length:]
                elif (int(signal_pct * len(signal)) == segment_length):
                    length_to_use = segment_length

                signal = signal[-length_to_use:]
                return signal

def add_signal_data_cwru(metadata_df: pd.DataFrame, fast_mode: bool = True, signal_pct: float = 1.0, end_portion: bool = False) -> pd.DataFrame:
    """This function reads the raw signal data from the CWRU dataset and adds it to the metadata DataFrame.

    Parameters:
    ----------
    metadata_df: pd.DataFrame
        Metadata DataFrame containing the information about the waveforms.

    Returns:
    -------
    waveforms_df: pd.DataFrame
        DataFrame containing the metadata information and the raw signal data
    """

    raw_data = []

    if fast_mode:
        raw_df = pd.read_pickle(
            "data/datasets/cwru/processed/full_dataset.pkl"
        )
        raw_df = raw_df[raw_df["waveform_id"].isin(metadata_df["waveform_id"])]
        raw_df["waveform_id"] = pd.Categorical(
            raw_df["waveform_id"], categories=metadata_df["waveform_id"], ordered=True
        )
        raw_df = raw_df.sort_values(by="waveform_id").reset_index(drop=True)[
            ["signal"]
        ]
    else:
        for _, waveform in metadata_df.iterrows():
            if waveform.fault_type == "Normal":
                save_name = f"Normal_{waveform.signal_location}_{str(waveform.fs)[:2]}"
            else:
                save_name = f"{waveform.waveform_id[:-4]}"
            raw_data.append(
                pd.read_pickle(
                    f"data/datasets/cwru/raw/{waveform.load} HP/{save_name}.bz2"
                )
            )
            raw_df = pd.DataFrame(raw_data).map(lambda x: x[0]).reset_index(drop=True)

    metadata_df = metadata_df.reset_index(drop=True)

    waveforms_df = pd.concat([metadata_df, raw_df], axis=1)

    if signal_pct < 1.0:

        waveforms_df["signal"] = waveforms_df.apply(
            lambda x: extract_1s_segments(x["signal"],signal_pct, end_portion, fs=x['fs']), axis=1
        )

    return waveforms_df


def add_signal_data_paderborn(
    metadata_df: pd.DataFrame,
    only_vibration: bool = True,
    fast_mode: bool = True,
    envelope: bool = False,
    band: list = [500, 10000],
    signal_pct: float = 1.0,
    end_portion: bool = False,
) -> pd.DataFrame:
    """This function reads the raw signal data from the Paderborn dataset and adds it to the metadata DataFrame.

    Parameters:
    ----------
    metadata_df: pd.DataFrame
        Metadata DataFrame containing the information about the waveforms.
    only_vibration: bool
        If True, only the vibration signal is loaded. If False, the vibration, force, current, speed and the temperature signals are loaded.
    fast_mode: bool
        If True, a preprocessed version of the dataset (which includes all signals) is loaded for faster access.
    envelope: bool
        If True, the spectrum envelope is used instead of the raw vibration signal.
    band: list
        Frequency band for the envelope signal.

    Returns:
    -------
    waveforms_df: pd.DataFrame
        DataFrame containing the metadata information and the raw signal data
    """

    raw_data = []
    if fast_mode:
        if not only_vibration:
            raise ValueError("fast_mode only works with only_vibration=True")
        
        if not envelope:
            raw_df = pd.read_pickle(
                "data/datasets/paderborn/processed/vibration_only/full_dataset.pkl"
            )
        else:
            raw_df = pd.read_pickle(
                "data/datasets/paderborn/processed/vibration_only/full_dataset_withEnvelope.pkl"
            )

            raw_df["vibration"] = raw_df[f"envelope/{band[0]}-{band[1]}"]


        raw_df = raw_df[raw_df["waveform_id"].isin(metadata_df["waveform_id"])]
        # Sort by waveform_id based on the order in metadata_df
        raw_df["waveform_id"] = pd.Categorical(
            raw_df["waveform_id"], categories=metadata_df["waveform_id"], ordered=True
        )
        raw_df = raw_df.sort_values(by="waveform_id")

    else:
        for _, waveform in metadata_df.iterrows():
            raw_data.append(
                pd.read_pickle(
                    f"data/datasets/paderborn/raw/{waveform.bearing_id}/{waveform.waveform_id}.bz2"
                ).iloc[0]
            )

        raw_df = pd.DataFrame(raw_data)

    if only_vibration and not fast_mode:
        raw_df = raw_df[["vibration", "fs_vibration"]]
        raw_df = raw_df.rename(columns={"fs_vibration": "fs"})
    elif only_vibration and fast_mode:
        raw_df = raw_df[["vibration", "fs"]]

    raw_df["waveform_id"] = metadata_df.waveform_id.values
    waveforms_df = metadata_df.merge(raw_df.drop(columns='fs'), on="waveform_id")

    if signal_pct < 1.0:

        waveforms_df["vibration"] = waveforms_df.apply(
            lambda x: extract_1s_segments(x["vibration"],signal_pct, end_portion, fs=x['fs']), axis=1
        )

    return waveforms_df


def add_signal_data_ottawa(
    metadata_df: pd.DataFrame,
    fast_mode: bool = True,
    signal_pct: float = 1.0,
    end_portion: bool = False,
) -> pd.DataFrame:
    """This function reads the raw signal data from the Ottawa dataset and adds it to the metadata DataFrame.

    Parameters:
    ----------
    metadata_df: pd.DataFrame
        Metadata DataFrame containing the information about the waveforms.
    fast_mode: bool
        If True, a preprocessed version of the dataset is loaded for faster access.
    signal_pct: float
        Percentage of the signal to be used. Default is 1.0 (100%).
    end_portion: bool
        If True, the end portion of the signal is used. If False, the beginning portion is used.

    Returns:
    -------
    waveforms_df: pd.DataFrame
        DataFrame containing the metadata information and the raw signal data
    """

    raw_data = []

    if fast_mode:
        raw_df = pd.read_pickle(
            "data/datasets/ottawa/processed/vibration_only/full_dataset.pkl"
        )
        raw_df = raw_df[raw_df["waveform_id"].isin(metadata_df["waveform_id"])]
        raw_df["waveform_id"] = pd.Categorical(
            raw_df["waveform_id"], categories=metadata_df["waveform_id"], ordered=True
        )
        raw_df = raw_df.sort_values(by="waveform_id").reset_index(drop=True)[
            ["vibration"]
        ]

    else:
        for _, waveform in metadata_df.iterrows():
            if waveform.fault_type == "Healthy":
                folder = "1_Healthy"
                abreviation = "H"
            elif waveform.fault_type == "Inner":
                folder = "2_Inner_Race_Faults"
                abreviation = "I"
            elif waveform.fault_type == "Outer":
                folder = "3_Outer_Race_Faults"
                abreviation = "O"
            elif waveform.fault_type == "Ball":
                folder = "4_Ball_Faults"
                abreviation = "B"
            elif waveform.fault_type == "Cage":
                folder = "5_Cage_Faults"
                abreviation = "C"

            save_name = (
                f"{folder}/{abreviation}_{waveform.bearing_id}_{waveform.severity}"
            )

            raw_data.append(
                pd.read_csv(f"data/datasets/ottawa/raw/{save_name}.csv")[
                    "Accelerometer"
                ].values
            )

        raw_df = pd.DataFrame({"vibration": raw_data}).reset_index(drop=True)

    metadata_df = metadata_df.reset_index(drop=True)
    waveforms_df = pd.concat([metadata_df, raw_df], axis=1)

    if signal_pct < 1.0:
        waveforms_df["vibration"] = waveforms_df["vibration"].apply(
            lambda x: x[: int(len(x) * signal_pct)]
            if not end_portion
            else x[-int(len(x) * signal_pct) :]
        )

    return waveforms_df

def add_signal_data_hust(metadata_df: pd.DataFrame, fast_mode: bool = True) -> pd.DataFrame:
    """This function reads the raw signal data from the hust dataset and adds it to the metadata DataFrame.

    Parameters:
    ----------
    metadata_df: pd.DataFrame
        Metadata DataFrame containing the information about the waveforms.
    fast_mode: bool
        If True, a preprocessed version of the dataset is loaded for faster access.

    Returns:
    -------
    waveforms_df: pd.DataFrame
        DataFrame containing the metadata information and the raw signal data
    """

    if fast_mode:
        full = pd.read_pickle("/data/bearing_datasets/hust/processed/full_dataset.pkl")
        return full[full["waveform_id"].isin(metadata_df["waveform_id"])]

    raw_df = pd.read_pickle("/data/bearing_datasets/hust/processed/full_dataset.pkl")
    raw_df = raw_df[raw_df["waveform_id"].isin(metadata_df["waveform_id"])]
    # Sort by waveform_id based on the order in metadata_df
    raw_df["waveform_id"] = pd.Categorical(
        raw_df["waveform_id"], categories=metadata_df["waveform_id"], ordered=True
    )
    raw_df = raw_df.sort_values(by="waveform_id").reset_index(drop=True)[["data"]]

    metadata_df = metadata_df.reset_index(drop=True)
    waveforms_df = pd.concat([metadata_df, raw_df], axis=1)

    return waveforms_df