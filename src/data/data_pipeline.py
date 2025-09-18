import pandas as pd
import numpy as np
from typing import List, Dict, Literal
from collections.abc import Callable
from hamilton.function_modifiers import config, extract_fields
from src.data.pydantic_models import BearingDataset
from src.data.custom_dataset import VibrationDataset
from torch.utils.data import DataLoader, ConcatDataset
import logging


def read_metadata_files(
    datasets: List[BearingDataset], logger: logging.Logger
) -> List[BearingDataset]:
    logger.info("Reading metadata files")
    for dataset in datasets:
        logger.debug(f"Reading metadata file for {dataset.name}")
        dataset.metadata = pd.read_pickle(dataset.metadata_file)
        logger.debug(
            f"Metadata file for {dataset.name} read successfully. Found {len(dataset.metadata)} samples"
        )

    return datasets


def split_data(
    read_metadata_files: List[BearingDataset],
    logger: logging.Logger,
) -> List[BearingDataset]:
    datasets = read_metadata_files

    logger.info("Splitting data")
    for dataset in datasets:
        logger.debug(
            f"Splitting data for {dataset.name}. Found {len(dataset.metadata)} samples"
        )
        dataset.train, dataset.validation, dataset.test = dataset.split_function(
            dataset.metadata
        )


        logger.debug(
            f"Data for {dataset.name} split successfully. Train: {len(dataset.train)}, Test: {len(dataset.test)}"
        )

    return datasets


@extract_fields(
    dict(
        X_train=pd.DataFrame,
        y_train=pd.Series,
        X_val=pd.DataFrame,
        y_val=pd.Series,
        X_test=pd.DataFrame,
        y_test=pd.Series,
    )
)
@config.when(pipeline="traditional")
def create_traditional_datasets(
    split_data: List[BearingDataset],
    features_list: List[str],
    feature_store_path: str,
) -> Dict[str, pd.DataFrame]:
    datasets = split_data

    feature_store = pd.read_pickle(feature_store_path)

    # Read features
    X_train = pd.concat(
        [
            feature_store[
                feature_store["waveform_id"].isin(dataset.train["waveform_id"])
            ][features_list]
            for dataset in datasets
        ]
    )
    X_val = pd.concat(
        [
            feature_store[
                feature_store["waveform_id"].isin(dataset.validation["waveform_id"])
            ][features_list]
            for dataset in datasets
        ]
    )
    X_test = pd.concat(
        [
            feature_store[
                feature_store["waveform_id"].isin(dataset.test["waveform_id"])
            ][features_list]
            for dataset in datasets
        ]
    )

    # Read labels
    y_train = pd.concat([dataset.train[dataset.label_column] for dataset in datasets])
    y_val = pd.concat(
        [dataset.validation[dataset.label_column] for dataset in datasets]
    )
    y_test = pd.concat([dataset.test[dataset.label_column] for dataset in datasets])

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
    }


@config.when(pipeline="deep_learning")
def add_signal_data(
    split_data: List[BearingDataset],
    train_pct: float = 1.0,
    test_pct: float = 1.0,
) -> List[BearingDataset]:
    datasets = split_data

    print("Adding signal data to datasets...")

    if train_pct < 1.0 or test_pct < 1.0:
        for dataset in datasets:
            dataset.train = dataset.add_signal_function(
                dataset.train, signal_pct=train_pct
            )


            if (dataset.validation is not None) and (len(dataset.validation) > 0):
                dataset.validation = dataset.add_signal_function(
                    dataset.validation, signal_pct=test_pct, end_portion=True
                )

            dataset.test = dataset.add_signal_function(
                dataset.test, signal_pct=test_pct, end_portion=True
            )
    else:
        for dataset in datasets:

            dataset.train = dataset.add_signal_function(dataset.train)

            if (dataset.validation is not None) and (
                len(dataset.validation) > 0
            ):  #  (len(dataset.validation) > 0) |
                dataset.validation = dataset.add_signal_function(dataset.validation)

            dataset.test = dataset.add_signal_function(dataset.test)

    return datasets


@config.when(pipeline="deep_learning")
def apply_normalization(
    add_signal_data: List[BearingDataset],
    normalization_function: Callable,
    normalization_strategy: str = "global",  # dataset-wise, entry-wise or global
) -> List[BearingDataset]:
    datasets = add_signal_data

    print("Applying normalization to datasets...")
    valid_strategies = ["dataset-wise", "entry-wise", "global", "none"]

    if normalization_strategy not in valid_strategies:
        raise ValueError(
            f"Invalid normalization strategy '{normalization_strategy}', please choose between 'entry-wise', 'dataset-wise', 'global' or 'none'"
        )

    # Skip normalization if normalization_strategy is none
    if normalization_strategy == "none":
        return datasets

    elif normalization_strategy == "global":
        train_data = np.concatenate(
            [
                np.concatenate(dataset.train[dataset.signal_column].values)
                if len(dataset.signal_column) == 1
                else np.concatenate(
                    dataset.train[dataset.signal_column].values.flatten()
                )
                for dataset in datasets
            ]
        )

        stats = {
            "min": np.min(train_data),
            "max": np.max(train_data),
            "mean": np.mean(train_data),
            "std": np.std(train_data),
        }

        norm_type = (
            "dataset-wise"  # global is the same as dataset-wise (using global stats)
        )
    else:
        norm_type = normalization_strategy
        stats = {}

    for dataset in datasets:
        if normalization_strategy == "dataset-wise":
            # Calculate train normalization stats (min, max, mean, std)
            train_data = np.concatenate(
                [
                    np.concatenate(dataset.train[dataset.signal_column].values)
                    if len(dataset.signal_column) == 1
                    else np.concatenate(
                        dataset.train[dataset.signal_column].values.flatten()
                    )
                    for dataset in datasets
                ]
            )
            stats = {
                "min": np.min(train_data),
                "max": np.max(train_data),
                "mean": np.mean(train_data),
                "std": np.std(train_data),
            }

        # Apply normalization to train, validation and test sets
        dataset.train[dataset.signal_column] = dataset.train[
            dataset.signal_column
        ].apply(lambda x: normalization_function(x, norm_type=norm_type, **stats))

        if (dataset.validation is not None) and (len(dataset.validation) > 0):
            dataset.validation[dataset.signal_column] = dataset.validation[
                dataset.signal_column
            ].apply(lambda x: normalization_function(x, norm_type=norm_type, **stats))

        dataset.test[dataset.signal_column] = dataset.test[dataset.signal_column].apply(
            lambda x: normalization_function(x, norm_type=norm_type, **stats)
        )

    return datasets


@extract_fields(
    dict(
        train_dataset=VibrationDataset,
        validation_dataset=VibrationDataset,
        test_dataset=VibrationDataset,
    )
)
@config.when(pipeline="deep_learning")
def create_datasets(
    apply_normalization: List[BearingDataset],
    transform: Callable,
    channels_output: int = 3,
    segment_length: int = 11500,
    segment_length_eval: int = 11500,
    overlap_pct: float | int = 0.97,
    overlap_pct_eval: float | int = 0.97,
    segmentation_strategy: Literal[
        "full_signal", "fixed_segments", "overlap"
    ] = "overlap",
    segmentation_strategy_eval: Literal[
        "full_signal", "fixed_segments", "overlap"
    ] = "overlap",
    dataset_multiplier: int = 1,
    augmentations: Dict[str, Callable] = None,
) -> Dict[str, VibrationDataset]:
    datasets = apply_normalization

    train_datasets = []
    validation_datasets = []
    test_datasets = []

    if augmentations is None:
        augmentations = {}

    pre_repr = augmentations.get("pre_repr", None)
    post_repr = augmentations.get("post_repr", None)

    if pre_repr is not None:
        train_augmentation_pre_repr_transform = pre_repr.get("train", None)
        test_augmentation_pre_repr_transform = pre_repr.get("test", None)
        validation_augmentation_pre_repr_transform = pre_repr.get("validation", None)
    else:
        (
            train_augmentation_pre_repr_transform,
            test_augmentation_pre_repr_transform,
            validation_augmentation_pre_repr_transform,
        ) = None, None, None

    if post_repr is not None:
        train_augmentation_post_repr_transform = post_repr.get("train", None)
        test_augmentation_post_repr_transform = post_repr.get("test", None)
        validation_augmentation_post_repr_transform = post_repr.get("validation", None)
    else:
        (
            train_augmentation_post_repr_transform,
            test_augmentation_post_repr_transform,
            validation_augmentation_post_repr_transform,
        ) = None, None, None

    for enum, dataset in enumerate(datasets):
        train_datasets.append(
            VibrationDataset(
                df=dataset.train,
                stage="train",
                transform=transform,
                channels_output=channels_output,
                segment_length=segment_length,
                overlap_pct=overlap_pct,
                label_names=dataset.label_column,
                signal_name=dataset.signal_column,
                segmentation_strategy=segmentation_strategy,
                dataset_multiplier=dataset_multiplier,
                preprocess_function=dataset.preprocess_function,
                augs_pre_repr_transform=train_augmentation_pre_repr_transform,
                augs_post_repr_transform=train_augmentation_post_repr_transform,
            )
        )

        test_datasets.append(
            VibrationDataset(
                dataset.test,
                stage="test",
                transform=transform,
                channels_output=channels_output,
                segment_length=segment_length_eval,
                overlap_pct=overlap_pct_eval,
                label_names=dataset.label_column,
                signal_name=dataset.signal_column,
                dataset_multiplier=1,
                segmentation_strategy=segmentation_strategy_eval,
                preprocess_function=dataset.preprocess_function,
                augs_pre_repr_transform=test_augmentation_pre_repr_transform,
                augs_post_repr_transform=test_augmentation_post_repr_transform,
            )
        )

        #print(f"Dataset {dataset.name} - Validation set size: {len(dataset.validation)}")
        if dataset.validation is not None:  # len(dataset.validation) > 0:
            validation_datasets.append(
                VibrationDataset(
                    dataset.validation,
                    stage="validation",
                    transform=transform,
                    channels_output=channels_output,
                    segment_length=segment_length_eval,
                    overlap_pct=overlap_pct_eval,
                    label_names=dataset.label_column,
                    signal_name=dataset.signal_column,
                    segmentation_strategy=segmentation_strategy_eval,
                    dataset_multiplier=1,
                    preprocess_function=dataset.preprocess_function,
                    augs_pre_repr_transform=validation_augmentation_pre_repr_transform,
                    augs_post_repr_transform=validation_augmentation_post_repr_transform,
                )
            )
        else:  # Set test as validation if no validation set
            #validation_datasets.append(test_datasets[enum])
            continue

    train_dataset = ConcatDataset(train_datasets)
    validation_dataset = ConcatDataset(validation_datasets) if len(validation_datasets) > 0 else None
    test_dataset = ConcatDataset(test_datasets)

    return {
        "train_dataset": train_dataset,
        "validation_dataset": validation_dataset,
        "test_dataset": test_dataset,
    }


@extract_fields(
    dict(
        train_dataloader=DataLoader,
        validation_dataloader=DataLoader,
        test_dataloader=DataLoader,
    )
)
@config.when(pipeline="deep_learning")
def setup_data_loaders(
    train_dataset: VibrationDataset,
    validation_dataset: VibrationDataset,
    test_dataset: VibrationDataset,
    batch_size: int,
    num_workers: int = 12,
    collate_train: Callable = None,
    collate_val: Callable = None,
) -> Dict[str, DataLoader]:
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=False,
        collate_fn=collate_train,
    )

    if validation_dataset is not None:
        validation_dataloader = DataLoader(
            validation_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=False,
            collate_fn=collate_val,
        )
    else:
        validation_dataloader = None

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        collate_fn=collate_val,
    )

    return {
        "train_dataloader": train_dataloader,
        "validation_dataloader": validation_dataloader,
        "test_dataloader": test_dataloader,
    }
