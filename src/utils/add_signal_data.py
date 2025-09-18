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


def add_signal_data_cwru(metadata_df: pd.DataFrame) -> pd.DataFrame:
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

    for _, waveform in metadata_df.iterrows():
        if waveform.fault_type == "Normal":
            save_name = f"Normal_{waveform.signal_location}_{str(waveform.fs)[:2]}"
        else:
            # save_name = f"{waveform.fault_type}_{waveform.fault_size}_{waveform.signal_location}_{str(waveform.fs)[:2]}"
            # save_name = f"{waveform.fault_type}_{waveform.fault_size}_{waveform.fault_location}_{waveform.signal_location}_{str(waveform.fs)[:2]}"
            save_name = f"{waveform.waveform_id[:-4]}"
        raw_data.append(
            pd.read_pickle(
                f"/data/bearing_datasets/cwru/raw/{waveform.load} HP/{save_name}.bz2"
            )
        )

    raw_df = pd.DataFrame(raw_data).map(lambda x: x.reshape(-1)).reset_index(drop=True)
    metadata_df = metadata_df.reset_index(drop=True)

    waveforms_df = pd.concat([metadata_df, raw_df], axis=1)

    return waveforms_df


def add_signal_data_paderborn(
    metadata_df: pd.DataFrame,
    only_vibration: bool = True,
    fast_mode: bool = True,
    resampled: bool = False,
    envelope: bool = False,
    band: list = [500, 10000],
) -> pd.DataFrame:
    """This function reads the raw signal data from the Paderborn dataset and adds it to the metadata DataFrame.

    Parameters:
    ----------
    metadata_df: pd.DataFrame
        Metadata DataFrame containing the information about the waveforms.
    only_vibration: bool
        If True, only the vibration signal is loaded. If False, the vibration, force, current, speed and the temperature signals are loaded.

    Returns:
    -------
    waveforms_df: pd.DataFrame
        DataFrame containing the metadata information and the raw signal data
    """

    raw_data = []
    if fast_mode:
        if not only_vibration:
            raise ValueError("fast_mode only works with only_vibration=True")
        if not resampled:
            if not envelope:
                raw_df = pd.read_pickle(
                    "/data/bearing_datasets/paderborn/processed/vibration_only/full_dataset.pkl"
                )
            else:
                raw_df = pd.read_pickle(
                    "/data/bearing_datasets/paderborn/processed/vibration_only/full_dataset_withEnvelope.pkl"
                )

                raw_df["vibration"] = raw_df[f"envelope/{band[0]}-{band[1]}"]
        else:
            raw_df = pd.read_pickle(
                "/data/bearing_datasets/paderborn/processed/vibration_only/resampled_dataset_20kHz.pkl"
            )

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
                    f"/data/bearing_datasets/paderborn/raw/{waveform.bearing_id}/{waveform.waveform_id}.bz2"
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

    return waveforms_df


def add_signal_data_lva(
    metadata_df: pd.DataFrame, fast_mode: bool = False
) -> pd.DataFrame:
    """This function reads the raw signal data from the LVA dataset and adds it to the metadata DataFrame.

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
        raw_df = pd.read_pickle("/data/bearing_datasets/lva/processed/full_dataset.pkl")
        raw_df = raw_df[raw_df["waveform_id"].isin(metadata_df["waveform_id"])]
        # Sort by waveform_id based on the order in metadata_df
        raw_df["waveform_id"] = pd.Categorical(
            raw_df["waveform_id"], categories=metadata_df["waveform_id"], ordered=True
        )
        raw_df = raw_df.sort_values(by="waveform_id").reset_index(drop=True)[["Signal"]]

    else:
        for _, waveform in metadata_df.iterrows():
            if (waveform["Condition"] == "normal") and (waveform["Setup"] != "d"):
                filename = f"normal_{waveform['RPM']}rpm_{waveform['Signal Location'].lower().replace(' ', '-')}.bz2"
            elif (waveform["Condition"] == "normal") and (waveform["Setup"] == "d"):
                filename = f"normal_{waveform['RPM']}rpm_{waveform['Unbalance Level']}__{waveform['Signal Location'].lower().replace(' ', '-')}.bz2"
            elif (waveform["Condition"] != "normal") and (waveform["Setup"] == "d"):
                filename = f"{waveform['Condition']}_{waveform['RPM']}rpm_{waveform['Unbalance Level']}_{waveform['Fault Location'].lower().replace(' ', '-')}_{waveform['Signal Location'].lower().replace(' ', '-')}.bz2"
            else:
                filename = f"{waveform['Condition']}_{waveform['RPM']}rpm_{waveform['Fault Location'].lower().replace(' ', '-')}_{waveform['Signal Location'].lower().replace(' ', '-')}.bz2"

            setup = waveform["Setup"]

            raw_data.append(
                pd.read_pickle(
                    f"/data/bearing_datasets/lva/raw/Setup {setup}/{filename}"
                )
            )

        raw_df = pd.DataFrame(raw_data).reset_index(drop=True)

    metadata_df = metadata_df.reset_index(drop=True)
    waveforms_df = pd.concat([metadata_df, raw_df], axis=1)

    return waveforms_df


def add_signal_data_ottawa(
    metadata_df: pd.DataFrame,
    fast_mode: bool = False,
    signal_pct: float = 1.0,
    end_portion: bool = False,
) -> pd.DataFrame:
    """This function reads the raw signal data from the Ottawa dataset and adds it to the metadata DataFrame.

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
            "/data/bearing_datasets/ottawa/processed/full_dataset.pkl"
        )
        raw_df = raw_df[raw_df["waveform_id"].isin(metadata_df["waveform_id"])]
        # Sort by waveform_id based on the order in metadata_df
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
                pd.read_csv(f"/data/bearing_datasets/ottawa/raw/{save_name}.csv")[
                    "Accelerometer"
                ].values
            )

        raw_df = pd.DataFrame({"vibration": raw_data}).reset_index(drop=True)

    metadata_df = metadata_df.reset_index(drop=True)
    waveforms_df = pd.concat([metadata_df, raw_df], axis=1)

    if signal_pct < 1.0:
        # selects a pertentage of the signal. if end_portion is True, it selects the last % portion of the signal, otherwise it selects the first portion.
        # each signal may have a different length, so we need to calculate the number of samples to select based on the signal length.
        waveforms_df["vibration"] = waveforms_df["vibration"].apply(
            lambda x: x[: int(len(x) * signal_pct)]
            if not end_portion
            else x[-int(len(x) * signal_pct) :]
        )

    return waveforms_df


def add_signal_to_synthetic_dataset(metadata_df):
    def generate_signal(bff, sff, fs=1e4, rpm=1800, duration=1, label=0):
        a = 0.12
        y = 0.085 * label

        bff = bff * rpm / 60

        n = int(fs * duration)
        t = np.arange(n) / fs
        z = np.random.normal(0, 1, n)

        faulty_signature = y * (
            np.cos(2 * np.pi * bff * t)
            + 0.5 * np.cos(4 * np.pi * bff * t)
            + 0.25 * np.cos(6 * np.pi * bff * t)
        )
        bearing_signature = a * (
            np.cos(2 * np.pi * sff * t)
            + 0.5 * np.cos(4 * np.pi * sff * t)
            + 0.25 * np.cos(6 * np.pi * sff * t)
        )

        signal = z + faulty_signature + bearing_signature
        signal -= np.mean(signal)  # Remove DC offset
        return signal

    metadata_df["signal"] = metadata_df.apply(
        lambda row: generate_signal(
            row["bff"],
            row["sf"],
            rpm=row["rpm"],
            duration=row["duration"],
            label=row["label"],
        ),
        axis=1,
    )

    return metadata_df


def add_signal_data_hust(metadata_df: pd.DataFrame) -> pd.DataFrame:
    """This function reads the raw signal data from the Ottawa dataset and adds it to the metadata DataFrame.

    Parameters:
    ----------
    metadata_df: pd.DataFrame
        Metadata DataFrame containing the information about the waveforms.

    Returns:
    -------
    waveforms_df: pd.DataFrame
        DataFrame containing the metadata information and the raw signal data
    """

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
