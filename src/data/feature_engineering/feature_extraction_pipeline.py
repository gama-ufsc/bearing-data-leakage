from functools import partial
from logging import Logger
from typing import Callable, List, Union

import pandas as pd
from sklearn.pipeline import Pipeline

from src.data.preprocess import (
    segment_dataset,
)
from src.data.pydantic_models import BearingDataset


def read_metadata_files(
    datasets: List[BearingDataset], logger: Logger
) -> List[BearingDataset]:
    logger.info("Reading metadata files")
    for dataset in datasets:
        logger.debug(f"Reading metadata file for {dataset.name}")
        dataset.metadata = pd.read_pickle(dataset.metadata_file)
        logger.debug(
            f"Metadata file for {dataset.name} read successfully. Found {len(dataset.metadata)} samples"
        )

    return datasets


def add_signal_data(
    read_metadata_files: List[BearingDataset],
) -> List[BearingDataset]:
    datasets = read_metadata_files

    for dataset in datasets:
        dataset.metadata = dataset.add_signal_function(dataset.metadata)

    return datasets


def preprocess_data(
    add_signal_data: List[BearingDataset],
    resample: bool,
    segment: bool,
    segment_size: int = 12000,
    overlap_pct: float = 0.0,
) -> List[BearingDataset]:
    datasets = add_signal_data

    for dataset in datasets:
        dataset.metadata = dataset.preprocess_function(
            dataset.metadata, resample=resample
        )

        if segment:
            print(f"Segmenting dataset {dataset.name}. Original size: {len(dataset.metadata)}")
            dataset.metadata = segment_dataset(
                dataset.metadata,
                signal_column=dataset.signal_column,
                segment_size=segment_size,
                overlap_pct = overlap_pct,
            ).reset_index(drop=True)
            print(f"Segmented dataset {dataset.name}. New size: {len(dataset.metadata)}")

    return datasets


def extract_features(
    preprocess_data: List[BearingDataset],
    feature_pipeline: Pipeline,
    cols_to_keep: List,
    logger: Logger,
) -> List[BearingDataset]:
    datasets = preprocess_data

    for dataset in datasets:
        logger.info(f"Extracting features for {dataset.name}")
        cols_to_drop = [
            column for column in dataset.metadata.columns if column not in cols_to_keep
        ]
        print(f"Dataset shape before feature extraction: {dataset.metadata.shape}")
        dataset.metadata = feature_pipeline.fit_transform(dataset.metadata)
        print(f"Dataset shape after feature extraction: {dataset.metadata.shape}")
        print(f"Dropping columns: {cols_to_drop}")
        dataset.metadata = dataset.metadata.drop(columns=cols_to_drop)

    return datasets


def save_on_feature_store(
    extract_features: List[BearingDataset],
    update_feature_store: Union[Callable, partial],
    logger: Logger,
    segment: bool = False,
) -> pd.DataFrame:
    datasets = extract_features

    for dataset in datasets:
        logger.info(f"Saving data to Feature Store for {dataset.name}")
        feature_store = update_feature_store(dataset.metadata, log=logger, seg=segment)
        logger.info(f"Data saved for {dataset.name}!")
        logger.info(f"feature_store shape: {feature_store.shape}")

    return feature_store
