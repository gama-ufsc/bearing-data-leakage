import pandas as pd


def update_feature_store(
    df: pd.DataFrame, log, file_path: str, backup_file_path: str, rewrite=False
) -> pd.DataFrame:
    """
    Updates the feature store with the new data.
    """

    if file_path.endswith(".pkl"):
        log.info(f"Reading feature store from: {file_path}")
        try:
            feature_store = pd.read_pickle(file_path)
        except FileNotFoundError:
            log.info("Feature store file not found. Creating a new one.")
            feature_store = pd.DataFrame(
                data=[{"waveform_id": ""}, {"waveform_id": ""}], columns=["waveform_id"]
            )
    else:
        log.info("File format not supported.")
        return

    if df["waveform_id"].isin(feature_store["waveform_id"].unique()).sum() != len(df):
        # Check new waveforms and update the feature store
        new_waveforms = df[~df["waveform_id"].isin(feature_store["waveform_id"])][
            "waveform_id"
        ].values

        # Find common columns between feature_store and df
        common_columns = feature_store.columns.intersection(df.columns)

        # Select only the common columns from df
        df_common = df[common_columns]

        # Reindex df_common to have the same columns as feature_store, filling missing ones with NaN
        df_reindexed = df_common.reindex(columns=feature_store.columns)

        # Concatenate feature_store and the subset of df that contains the new waveforms
        result_df = pd.concat(
            [
                feature_store,
                df_reindexed[df_reindexed["waveform_id"].isin(new_waveforms)],
            ],
            ignore_index=True,
        )

        feature_store = result_df.drop_duplicates("waveform_id").reset_index(drop=True)

    different_columns = set(df.columns) - set(feature_store.columns)

    if rewrite:
        for column in df.columns:
            if column != "waveform_id":
                if column in feature_store.columns:
                    feature_store_map = feature_store.set_index("waveform_id")[
                        column
                    ].to_dict()
                else:
                    feature_store_map = {}
                df_column_map = df.set_index("waveform_id")[column].to_dict()
                feature_store_map.update(df_column_map)
                feature_store[column] = feature_store["waveform_id"].map(
                    feature_store_map
                )
    else:
        for column in different_columns:
            column_map = df.set_index("waveform_id")[column].to_dict()
            feature_store[column] = feature_store["waveform_id"].map(column_map)

    try:
        feature_store.to_pickle(file_path)
        # feature_store.to_parquet(file_path, compression="gzip")
    except Exception as e:
        print(e)
        log.error("Error writing feature store to disk")
        log.info("Saving feature store to a backup file")
        # feature_store.to_parquet(backup_file_path, compression="gzip")
        feature_store.to_pickle(backup_file_path)
        log.info("Feature store saved to backup file")

    return feature_store
