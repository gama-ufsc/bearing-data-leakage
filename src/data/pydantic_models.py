from pydantic import BaseModel, computed_field, ConfigDict
from typing import Optional, Literal, Union, List
from collections.abc import Callable
import pandas as pd

# Change here to add a new dataset
AllowedDataset = Literal["CWRU", "Paderborn", "LVA", "Ottawa", "Synthetic"]
DATASET_METADATA_PATHS = {
    "CWRU": "/data/bearing_datasets/cwru/processed/files_metadata.bz2",
    "Paderborn": "/data/bearing_datasets/paderborn/processed/files_metadata.bz2",
    "LVA": "/data/bearing_datasets/lva/processed/files_metadata.bz2",
    "Ottawa": "/data/bearing_datasets/ottawa/processed/files_metadata.bz2",
    # "Generic": "/data/bearing_datasets/generic/processed/files_3.bz2",
    "Synthetic": "/data/bearing_datasets/synthetic/metadata_df.bz2",
}


class BearingDataset(BaseModel):
    # Pydantic v2 configuration: allow arbitrary types (e.g., pd.DataFrame)
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: AllowedDataset
    add_signal_function: Callable
    signal_column: str
    label_column: Union[str, List[str]]
    split_function: Optional[Callable] = None
    preprocess_function: Optional[Callable] = None
    metadata: Optional[pd.DataFrame] = None
    train: Optional[pd.DataFrame] = None
    validation: Optional[pd.DataFrame] = None
    test: Optional[pd.DataFrame] = None

    # Fill metadata_file based on the input from AllowedDataset
    @computed_field
    @property
    def metadata_file(self) -> str:
        try:
            return DATASET_METADATA_PATHS[self.name]
        except KeyError:
            raise ValueError(
                f"Dataset {self.name} not added on the DATASET_METADATA_PATHS, please choose from {list(DATASET_METADATA_PATHS.keys())}"
            )

    def __str__(self):
        if self.metadata is None:
            return f"{self.name} dataset without metadata"
        else:
            return f"{self.name} dataset with {len(self.metadata)} samples"

    def __repr__(self):
        return self.__str__()
