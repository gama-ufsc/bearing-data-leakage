import random
from itertools import combinations, product
from typing import Literal, Tuple
from sklearn.model_selection import GroupShuffleSplit

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# ------------------ Global variables  ------------------

# Real fault bearings - Paderborn dataset
HEALTHY_BEARING_IDS = ["K001", "K002", "K003", "K004", "K005", "K006"]
OUTER_BEARING_IDS = ["KA04", "KA15", "KA16", "KA22", "KA30"]
INNER_BEARING_IDS = ["KI04", "KI14", "KI16", "KI18", "KI21", "KI17"]

# Remaining real fault and normal condition bearings - Paderborn dataset
COMBINATION_BEARINGS_IDS = ["KB23", "KB24", "KB27"]

# Artificially damaged fault bearings - Paderborn dataset
ARTIFICIAL_DAMAGED_BEARINGS_IDS = [
    "KA01",
    "KA03",
    "KA05",
    "KA06",
    "KA07",
    "KA08",
    "KA09",
    "KI01",
    "KI03",
    "KI05",
    "KI07",
    "KI08",
]

# Following the rolling bearing table from the paper
# we have 3 columns: healthy, outer, inner
bearing_table = np.array(
    [
        [healthy, outer, inner]
        for healthy, outer, inner in zip(
            HEALTHY_BEARING_IDS, OUTER_BEARING_IDS, INNER_BEARING_IDS
        )
    ]
)

# Fault bearing operating conditions
# each condition is a row, where the columns are:
# [rotational_speed, load_torque, radial_force]
operating_conditions = np.array(
    [[1500, 0.7, 1000], [900, 0.7, 1000], [1500, 0.1, 1000], [1500, 0.7, 400]]
)

### Ottawa proposed split

def ottawa_proposed_split(
    df,
    n_bearings_per_fault_mode=2,
    run=0,
    random_state=42,
    debug=False,
    return_ids: bool = False,
    no_val: bool = False,
):
    """
    Generate custom CV splits where each test set contains 2 IDs from each fault type.

    Parameters:
    - df: DataFrame with at least four columns representing class-wise IDs.
    - n_bearings_per_fault_mode: Number of bearings per fault mode in the test set.
    - run: Run number for reproducibility.
    - debug: Print debug info.

    Returns:
    - train, val, test: DataFrames for training, validation, and test sets.
    """

    # Get unique IDs per fault type
    inner_ids = [1, 2, 3, 4, 5]
    outer_ids = [6, 7, 8, 9, 10]
    ball_ids = [11, 12, 13, 14, 15]
    cage_ids = [16, 17, 18, 19, 20]

    # All 2-combinations for each
    inner_combos = list(combinations(inner_ids, n_bearings_per_fault_mode))
    outer_combos = list(combinations(outer_ids, n_bearings_per_fault_mode))
    ball_combos = list(combinations(ball_ids, n_bearings_per_fault_mode))
    cage_combos = list(combinations(cage_ids, n_bearings_per_fault_mode))

    # Cartesian product to form full test sets (can be huge!)
    all_combos = list(product(inner_combos, outer_combos, ball_combos, cage_combos))

    # Shuffle
    rng = random.Random(random_state)
    rng.shuffle(all_combos)

    # Limit to n_splits
    selected_combos = all_combos[run]
    print(f"Selected test IDs: {selected_combos}")

    inner_test, outer_test, ball_test, cage_test = selected_combos
    # Combine all test IDs into one set
    test_ids = set(inner_test) | set(outer_test) | set(ball_test) | set(cage_test)
    train_ids = list(set(df["bearing_id"].astype(int).unique()) - test_ids)
    test_mask = df["bearing_id"].astype(int).isin(test_ids)

    test = df[test_mask]
    train = df[~test_mask]

    if debug:
        print(f"Inner Test IDs: {inner_test}")
        print(f"Outer Test IDs: {outer_test}")
        print(f"Ball  Test IDs: {ball_test}")
        print(f"Cage  Test IDs: {cage_test}")
        print(f"Length of train: {len(train)}")
        print(f"Length of test: {len(test)}")

    if no_val:
        return train, None, test
    val = test

    if not return_ids:
        return train, val, test
    else:
        return train_ids, list(test_ids)


def ottawa_custom_split(
        df,
        n_bearings_per_fault_mode=2,
        run=0,
        random_state=42,
        no_val=False,
    ):
    """
    Generate custom CV splits where each test set contains 2 IDs from each fault type.
    """

    inner_ids = [1, 2, 3, 4, 5]
    outer_ids = [6, 7, 8, 9, 10]
    ball_ids = [11, 12, 13, 14, 15]
    cage_ids = [16, 17, 18, 19, 20]

    inner_combos = list(combinations(inner_ids, 4))
    outer_combos = list(combinations(outer_ids, 4))
    ball_combos = list(combinations(ball_ids, 4))
    cage_combos = list(combinations(cage_ids, 4))

    all_combos = list(product(inner_combos, outer_combos, ball_combos, cage_combos))

    rng = random.Random(random_state)
    rng.shuffle(all_combos)
    selected_combos = all_combos[run]
    inner, outer, ball, cage = selected_combos

    df['bearing_id'] = df['bearing_id'].astype(int)

    normal_bearings = list(set(inner_ids + outer_ids + ball_ids + cage_ids) - set(inner) - set(outer) - set(ball) - set(cage))
    print(f"Bearings selected to be normal: " f"{normal_bearings}")

    df = df[~(df['bearing_id'].isin(normal_bearings) & (df['fault_type'] != 'Healthy'))]
    df = df[~(df['bearing_id'].isin(inner) & (df['fault_type'] != 'Inner'))]
    df = df[~(df['bearing_id'].isin(outer) & (df['fault_type'] != 'Outer'))]
    df = df[~(df['bearing_id'].isin(ball) & (df['fault_type'] != 'Ball'))]
    df = df[~(df['bearing_id'].isin(cage) & (df['fault_type'] != 'Cage'))]

    normal_combos = list(combinations(normal_bearings, n_bearings_per_fault_mode))
    inner_combos = list(combinations(inner, n_bearings_per_fault_mode))
    outer_combos = list(combinations(outer, n_bearings_per_fault_mode))
    ball_combos = list(combinations(ball, n_bearings_per_fault_mode))
    cage_combos = list(combinations(cage, n_bearings_per_fault_mode))

    # Cartesian product to form full test sets (can be huge!)
    all_combos = list(product(normal_combos, inner_combos, outer_combos, ball_combos, cage_combos))

    rng.shuffle(all_combos)
    selected_combos = all_combos[run]
    print(f"Selected test IDs: {selected_combos}")

    normal_test, inner_test, outer_test, ball_test, cage_test = selected_combos
    test_ids = set(normal_test) | set(inner_test) | set(outer_test) | set(ball_test) | set(cage_test)

    test_mask = df["bearing_id"].isin(test_ids)
    test = df[test_mask]
    train = df[~test_mask]

    if no_val:
        return train, None, test
    val = test

    return train, val, test

def ottawa_leakage(
    df,
    n_bearings_per_fault_mode=2,
    run=0,
    random_state=42,
    debug=False,
    easy_split=True,
    segment_leakage=False,
):
    print(f"Ottawa Leakage Split - Easy split: {easy_split}")

    inner_ids = [1, 2, 3, 4, 5]
    outer_ids = [6, 7, 8, 9, 10]
    ball_ids = [11, 12, 13, 14, 15]
    cage_ids = [16, 17, 18, 19, 20]

    inner_combos = list(combinations(inner_ids, n_bearings_per_fault_mode))
    outer_combos = list(combinations(outer_ids, n_bearings_per_fault_mode))
    ball_combos = list(combinations(ball_ids, n_bearings_per_fault_mode))
    cage_combos = list(combinations(cage_ids, n_bearings_per_fault_mode))

    all_combos = list(product(inner_combos, outer_combos, ball_combos, cage_combos))

    rng = random.Random(random_state)
    rng.shuffle(all_combos)

    selected_combos = all_combos[run]
    print(f"Selected test IDs: {selected_combos}")

    inner_test, outer_test, ball_test, cage_test = selected_combos
    test_ids = set(inner_test) | set(outer_test) | set(ball_test) | set(cage_test)
    test_mask = df["bearing_id"].astype(int).isin(test_ids)

    test = df[test_mask]
    train = df[~test_mask]

    train.loc[:, "severity"] = train.severity.apply(lambda x: int(x))
    severity_2_ids = train[(train.severity == 2)]["waveform_id"].unique()

    train = train[~train["waveform_id"].isin(severity_2_ids)].reset_index(drop=True)

    # if easy_split:
    #    test = pd.concat([test, df[df["waveform_id"].isin(severity_2_ids)]], ignore_index=True).reset_index(drop=True)
    # objetivo: manter o tamanho de teste igual, retirando sinais de severidade 2 e adicionando parte no teste
    if easy_split:
        print(test.shape)
        test.loc[:, "severity"] = test.severity.apply(lambda x: int(x))
        test = test[~(test.severity == 2)]  # retira 8 sinais
        sev2 = (
            df[df["waveform_id"].isin(severity_2_ids)].sample(8, random_state=42).copy()
        )
        test = pd.concat([test, sev2], ignore_index=True).reset_index(drop=True)
    elif segment_leakage:
        # If segment leakage is enabled, keep all train bearings in test set (exclude test bearings)
        test = train

    val = test

    print(f"Test shape: {test.shape}")

    return train, val, test

#unused
# def diversity_ottawa(
#     df,
#     n_bearings_per_fault_mode=1,
#     run=0,
#     random_state=42,
#     debug=False,
#     split: Literal["1x1", "2x1", "3x1", "4x1"] = "4x1",
#     return_ids: bool = False,
# ):
#     """
#     Generate custom CV splits where each test set contains 1 ID from each fault type.

#     Parameters:
#     - df: DataFrame with at least four columns representing class-wise IDs.
#     - n_bearings_per_fault_mode: Number of bearings per fault mode in the test set.
#     - run: Run number for reproducibility.
#     - debug: Print debug info.
#     - split: Split configuration.
#     Returns:
#     - train, val, test: DataFrames for training, validation, and test sets.
#     """

#     # Get unique IDs per fault type
#     print(f"Diversity split on Ottawa: {split}")
#     inner_ids = [1, 2, 3, 4, 5]
#     outer_ids = [6, 7, 8, 9, 10]
#     ball_ids = [11, 12, 13, 14, 15]
#     cage_ids = [16, 17, 18, 19, 20]

#     # All 2-combinations for each
#     inner_combos = list(combinations(inner_ids, n_bearings_per_fault_mode))
#     outer_combos = list(combinations(outer_ids, n_bearings_per_fault_mode))
#     ball_combos = list(combinations(ball_ids, n_bearings_per_fault_mode))
#     cage_combos = list(combinations(cage_ids, n_bearings_per_fault_mode))

#     # Cartesian product to form full test sets (can be huge!)
#     all_combos = list(product(inner_combos, outer_combos, ball_combos, cage_combos))

#     # Shuffle
#     rng = random.Random(random_state)
#     rng.shuffle(all_combos)

#     # Limit to n_splits
#     selected_combos = all_combos[run]
#     print(f"Selected test IDs: {selected_combos}")

#     inner_test, outer_test, ball_test, cage_test = selected_combos
#     test_ids = set(inner_test) | set(outer_test) | set(ball_test) | set(cage_test)

#     outer_train_ids = list(set(outer_ids) - set(outer_test))
#     inner_train_ids = list(set(inner_ids) - set(inner_test))
#     ball_train_ids = list(set(ball_ids) - set(ball_test))
#     cage_train_ids = list(set(cage_ids) - set(cage_test))

#     if split == "4x1":
#         train_ids = list(set(df["bearing_id"].astype(int).unique()) - test_ids)
#     elif (split != "4x1") & (split in ["3x1", "2x1", "1x1"]):
#         random.seed(random_state)
#         inner_drop = rng.choice(list(inner_train_ids))
#         outer_drop = rng.choice(list(outer_train_ids))
#         ball_drop = rng.choice(list(ball_train_ids))
#         cage_drop = rng.choice(list(cage_train_ids))

#         inner_train_ids.remove(inner_drop)
#         outer_train_ids.remove(outer_drop)
#         ball_train_ids.remove(ball_drop)
#         cage_train_ids.remove(cage_drop)

#         if split != "3x1":
#             inner_drop = rng.choice(list(inner_train_ids))
#             outer_drop = rng.choice(list(outer_train_ids))
#             ball_drop = rng.choice(list(ball_train_ids))
#             cage_drop = rng.choice(list(cage_train_ids))

#             inner_train_ids.remove(inner_drop)
#             outer_train_ids.remove(outer_drop)
#             ball_train_ids.remove(ball_drop)
#             cage_train_ids.remove(cage_drop)

#             if split != "2x1":
#                 inner_drop = rng.choice(list(inner_train_ids))
#                 outer_drop = rng.choice(list(outer_train_ids))
#                 ball_drop = rng.choice(list(ball_train_ids))
#                 cage_drop = rng.choice(list(cage_train_ids))

#                 inner_train_ids.remove(inner_drop)
#                 outer_train_ids.remove(outer_drop)
#                 ball_train_ids.remove(ball_drop)
#                 cage_train_ids.remove(cage_drop)
        
#         train_ids = inner_train_ids + outer_train_ids + ball_train_ids + cage_train_ids
#     else:
#         raise ValueError("Invalid split value. Must be one of '1x1', '2x1', '3x1' or '4x1'.")
        
#     test_mask = df["bearing_id"].astype(int).isin(test_ids)

#     test = df[test_mask]
#     train = df[df.bearing_id.astype(int).isin(train_ids)]

#     if debug:
#         print(f"Inner Test IDs: {inner_test}")
#         print(f"Outer Test IDs: {outer_test}")
#         print(f"Ball  Test IDs: {ball_test}")
#         print(f"Cage  Test IDs: {cage_test}")
#         print(f"Length of train: {len(train)}")
#         print(f"Length of test: {len(test)}")

#     val = test

#     if not return_ids:
#         return train, val, test
#     else:
#         return train_ids, list(test_ids)


def cwru_optimization_split(
    df: pd.DataFrame,
    test_fold: Literal[1, 2, 3],
    train_side: Literal["FE", "DE"],
    withHP0: bool = True,
    train_path=None,
    test_path=None,
    **kwargs,
):
    """
    Split the CWRU dataset into training and test sets based on the test fold and train side.

    Args:
        df (pd.DataFrame): DataFrame containing the metadata.
        test_fold (int): Test fold number (1, 2, or 3).
        train_side (str): Side of the bearing to use for training ("FE" or "DE").

    Returns:
        Tuple(pd.DataFrame, pd.DataFrame): Train and test DataFrames.
    """

    if test_fold not in [1, 2, 3]:
        raise ValueError("Test fold must be 1, 2, or 3")

    if train_side not in ["FE", "DE", "both"]:
        raise ValueError("Train side must be 'FE' or 'DE' or 'both'")

    if (train_path is None) and (test_path is None):
        if not withHP0:
            train_path = (
                f"data/splits/cwru/fold_{test_fold}/train_{train_side}_noHP0.csv"
            )
            test_path = f"data/splits/cwru/fold_{test_fold}/test_{train_side}_noHP0.csv"
        else:
            train_path = f"data/splits/cwru/fold_{test_fold}/train_{train_side}.csv"
            test_path = f"data/splits/cwru/fold_{test_fold}/test_{train_side}.csv"

    train_ids = pd.read_csv(train_path)["waveform_id"].tolist()
    test_ids = pd.read_csv(test_path)["waveform_id"].tolist()

    train = df[df["waveform_id"].isin(train_ids)].reset_index(drop=True)
    test = df[df["waveform_id"].isin(test_ids)].reset_index(drop=True)

    val = test

    return train, val, test


def split_cwru_proposed(metadata_df, run=0, normal_train_side="DE", random_state=42, split_ratio: Literal["1x2", "2x1"] = "2x1", no_val=True):
     
    # Create combinations of sizes for test set. Each test set will contain 1 element from each list
    if split_ratio == "2x1":
        de_or = [["DE_OR_7"], ["DE_OR_14"], ["DE_OR_21"]]
        fe_or  = [["FE_OR_7"], ["FE_OR_14"], ["FE_OR_21"]]
        de_b = [["DE_B_7"], ["DE_B_14"], ["DE_B_21"]]
        fe_b = [["FE_B_7"], ["FE_B_14"], ["FE_B_21"]]
        de_ir = [["DE_IR_7"], ["DE_IR_14"], ["DE_IR_21"]]
        fe_ir = [["FE_IR_7"], ["FE_IR_14"], ["FE_IR_21"]]
    elif split_ratio == "1x2":
        de_or = [["DE_OR_7", "DE_OR_14"], ["DE_OR_7", "DE_OR_21"], ["DE_OR_14", "DE_OR_21"]]
        fe_or  = [["FE_OR_7", "FE_OR_14"], ["FE_OR_7", "FE_OR_21"], ["FE_OR_14", "FE_OR_21"]]
        de_b = [["DE_B_7", "DE_B_14"], ["DE_B_7", "DE_B_21"], ["DE_B_14", "DE_B_21"]]
        fe_b = [["FE_B_7", "FE_B_14"], ["FE_B_7", "FE_B_21"], ["FE_B_14", "FE_B_21"]]
        de_ir = [["DE_IR_7", "DE_IR_14"], ["DE_IR_7", "DE_IR_21"], ["DE_IR_14", "DE_IR_21"]]
        fe_ir = [["FE_IR_7", "FE_IR_14"], ["FE_IR_7", "FE_IR_21"], ["FE_IR_14", "FE_IR_21"]]
    else:
        raise ValueError("split_ratio must be either '1x2' or '2x1'")
    
    test_combinations = list(product(de_or, de_b, de_ir, fe_or, fe_b, fe_ir))
    random.seed(random_state)
    random.shuffle(test_combinations)
    
    test_groups = np.array(test_combinations[run]).flatten()

    df_filtered_ids = pd.read_pickle("data/splits/cwru/filtered_cwru.pkl")
    metadata_df = metadata_df[metadata_df["waveform_id"].isin(df_filtered_ids["waveform_id"].tolist())]

    metadata_df.loc[:, "fault_type"] = metadata_df["fault_type"].str.replace("OR@3", "OR")
    metadata_df.loc[:, "fault_type"] = metadata_df["fault_type"].str.replace("OR@6", "OR")
    metadata_df = metadata_df.copy() # pandas warning fix
    metadata_df["group"] = (
        metadata_df["fault_location"].astype(str)
        + "_"
        + metadata_df["fault_type"].astype(str)
        + "_"
        + metadata_df["fault_size"].astype(str)
    )

    df_train = metadata_df[~metadata_df.group.isin(test_groups)].reset_index(drop=True)
    df_test = metadata_df[metadata_df.group.isin(test_groups)].reset_index(drop=True)

    if normal_train_side == "DE":
        df_train = df_train[
            ~(
                (df_train["signal_location"] == "FE")
                & (df_train["fault_location"] == "DE")
            )
        ]
        df_test = df_test[
            ~(
                (df_test["signal_location"] == "DE")
                & (df_test["fault_location"] == "FE")
            )
        ]
    elif normal_train_side == "FE":
        df_train = df_train[
            ~(
                (df_train["signal_location"] == "DE")
                & (df_train["fault_location"] == "FE")
            )
        ]
        df_test = df_test[
            ~(
                (df_test["signal_location"] == "FE")
                & (df_test["fault_location"] == "DE")
            )
        ]
    elif normal_train_side == "both":
        pass

    # check if any split is equal to a tuning split. if so, we will recreate it with another seed
    tuning_fold1 = pd.read_parquet(f"data/splits/cwru/tuning/fold_1/test_{normal_train_side}.parquet")
    tuning_fold2= pd.read_parquet(f"data/splits/cwru/tuning/fold_2/test_{normal_train_side}.parquet")
    tuning_fold3 = pd.read_parquet(f"data/splits/cwru/tuning/fold_3/test_{normal_train_side}.parquet")

    test_ids = df_test.waveform_id.unique()
    tuning_fold1_ids = tuning_fold1.waveform_id.unique()
    tuning_fold2_ids = tuning_fold2.waveform_id.unique()
    tuning_fold3_ids = tuning_fold3.waveform_id.unique()

    recreate = False
    if (set(test_ids) == set(tuning_fold1_ids)) or (set(test_ids) == set(tuning_fold2_ids)) or (set(test_ids) == set(tuning_fold3_ids)):
        print(f"Test set on run {run} matches one of the tuning folds. Recreating split with another seed.")
        recreate = True

    while recreate:

        _, _, test= split_cwru_proposed(metadata_df, run=run+200, normal_train_side=normal_train_side, random_state=random_state+1, split_ratio=split_ratio)
        test_ids = test.waveform_id.unique()

        if not((set(test_ids) == set(tuning_fold1_ids)) or (set(test_ids) == set(tuning_fold2_ids)) or (set(test_ids) == set(tuning_fold3_ids))):
            recreate = False
            print(f"New test set on run {run+200} does not match any of the tuning folds. Split created successfully.")
        df_test = test    

    if no_val:
        return df_train.reset_index(drop=True), None, df_test.reset_index(drop=True)
    else:
        val = df_test.copy()
        return df_train.reset_index(drop=True), val.reset_index(drop=True), df_test.reset_index(drop=True)

def cwru_leakage(df, train_group: str, leakage_split: Literal["condition", "segment", "no_leakage"], return_groups=False, **kwargs):

    def get_group(row):

        if row["signal_location"] == "DE" and row["fault_location"] == "DE":
            if int(row["fault_size"]) == 7:
                if int(row["load"]) == 0:
                    return "A"
                elif int(row["load"]) == 1:
                    return "B"
                elif int(row["load"]) == 2:
                    return "C"
                elif int(row["load"]) == 3:
                    return "D"
            elif int(row["fault_size"]) == 14:
                if int(row["load"]) == 0:
                    return "E"
                elif int(row["load"]) == 1:
                    return "F"
                elif int(row["load"]) == 2:
                    return "G"
                elif int(row["load"]) == 3:
                    return "H"
            elif int(row["fault_size"]) == 21:
                if int(row["load"]) == 0:
                    return "I"
                elif int(row["load"]) == 1:
                    return "J"
                elif int(row["load"]) == 2:
                    return "K"
                elif int(row["load"]) == 3:
                    return "L"
        else:
            return "Z"

    df_filtered_ids = pd.read_pickle("data/splits/cwru/filtered_cwru.pkl")
    df = df[df["waveform_id"].isin(df_filtered_ids["waveform_id"].tolist())]

    df['group'] = df.apply(lambda row: get_group(row), axis=1)

    row1 = ["A", "B", "C", "D"]
    row2 = ["E", "F", "G", "H"]
    row3 = ["I", "J", "K", "L"]
    
    if leakage_split == "condition":
        # First, check which row the train_group belongs to; then assign all remaining groups in that row as test_group
        if train_group in row1:
            test_group = row1.copy()
            test_group.remove(train_group)
        elif train_group in row2:
            test_group = row2.copy()
            test_group.remove(train_group)
        elif train_group in row3:
            test_group = row3.copy()
            test_group.remove(train_group)
    elif leakage_split == "segment":

        test_group = [train_group]
    elif leakage_split == "no_leakage":
        # The test groups will correspond to rows that are different from the train_group row
        if train_group in row1:
            test_group = row2 + row3
        elif train_group in row2:
            test_group = row1 + row3
        elif train_group in row3:
            test_group = row1 + row2
    else:
        raise ValueError("leakage_split must be either 'condition', 'segment' or 'no_leakage'")
            
    print(f"Train group: {train_group}, Test groups: {test_group}")
    train= df[df["group"] == train_group].copy()
    test = df[df["group"].isin(test_group)].copy()

    train["multiclass_label"] = (train["multiclass_label"] - 1).astype(int)
    test["multiclass_label"] = (test["multiclass_label"] - 1).astype(int)
    val=test



    if return_groups:
        return train_group, test_group
    return train, val, test

# Sampler for Paderborn
class CombinationSampler:
    def __init__(self, list1, list2, list3, random_seed=42):
        """
        Sampler constructor to generate all possible combinations

        Args:
            list1 (list): list of elements
            list2 (list): list of elements
            list3 (list): list of elements
            random_seed (int): random seed
        """
        self.combinations = list(product(list1, list2, list3))
        random.seed(random_seed)
        random.shuffle(self.combinations)

    def get_combination(self, run: int):
        """
        Random combination selection

        Params:
            run (int): run number

        Returns:
            Tuple: random combination
        """
        if not self.combinations:
            raise ValueError("No more combinations available")

        # Select the combination for the current run
        combination = self.combinations[run]

        return combination


list1 = [
    ["K001", "K003"],
    ["K001", "K002"],
    ["K001", "K004"],
    ["K001", "K005"],
    ["K001", "K006"],
    ["K002", "K003"],
    ["K002", "K004"],
    ["K002", "K005"],
    ["K002", "K006"],
    ["K003", "K004"],
    ["K003", "K005"],
    ["K003", "K006"],
    ["K004", "K005"],
    ["K004", "K006"],
    ["K005", "K006"],
]

list2 = [
    ["KA04", "KA16"],
    ["KA04", "KA15"],
    ["KA04", "KA22"],
    ["KA04", "KA30"],
    ["KA15", "KA16"],
    ["KA15", "KA22"],
    ["KA15", "KA30"],
    ["KA16", "KA22"],
    ["KA16", "KA30"],
    ["KA22", "KA30"],
]

list3 = [
    ["KI04", "KI16"],
    ["KI04", "KI14"],
    ["KI04", "KI21"],
    ["KI04", "KI18"],
    ["KI04", "KI17"],
    ["KI14", "KI16"],
    ["KI14", "KI18"],
    ["KI14", "KI21"],
    ["KI14", "KI17"],
    ["KI16", "KI18"],
    ["KI16", "KI21"],
    ["KI16", "KI17"],
    ["KI18", "KI21"],
    ["KI18", "KI17"],
    ["KI21", "KI17"],
]

SAMPLER = CombinationSampler(list1, list2, list3)

### Paderborn proposed split
def train_test_split_proposed(
    df: pd.DataFrame,
    sampler: CombinationSampler,
    run: int,
    healthy_bearing_ids: list,
    outer_bearing_ids: list,
    inner_bearing_ids: list,
    use_artificial: bool = False,
    use_combined: bool = False,
    return_ids: bool = False,
    on_train: bool = False,
    no_val: bool = False,
):
    """
    Split the bearings into train, validation and test sets

    Args:
        df (pd.DataFrame): dataframe with the metadata
        sampler (CombinationSampler): sampler instance
        run (int): run number
        healthy_bearing_ids (list): healthy bearing ids
        outer_bearing_ids (list): outer bearing ids
        inner_bearing_ids (list): inner bearing ids

    Returns:
        Tuple(list): train, val and test sets
    """
    combination = sampler.get_combination(run=run)

    train_ids = (
        list(set(healthy_bearing_ids).difference(set(combination[0])))
        + list(set(outer_bearing_ids).difference(set(combination[1])))
        + list(set(inner_bearing_ids).difference(set(combination[2])))
    )

    if use_artificial:
        artificial_damaged_bearings_ids = [
            "KA01",
            "KA03",
            "KA05",
            "KA06",
            "KA07",
            "KA08",
            "KA09",
            "KI01",
            "KI03",
            "KI05",
            "KI07",
            "KI08",
        ]
        print(f"Using artificially damaged bearings: {artificial_damaged_bearings_ids}")
        train_ids += artificial_damaged_bearings_ids

    test_ids = list(combination[0] + combination[1] + combination[2])
    

    if use_combined:
        combined_bearings_ids = ["KB23", "KB24", "KB27"]
        if on_train:
            print(f"Using combined bearings on train: {combined_bearings_ids}")
            train_ids += combined_bearings_ids
        else:
            print(f"Using combined bearings on test: {combined_bearings_ids}")
            test_ids += combined_bearings_ids

    train = df[df["bearing_id"].isin(train_ids)]
    test = df[df["bearing_id"].isin(test_ids)]

    val = test

    print(f"Test bearings: {test['bearing_id'].unique()}")

    if return_ids:
        return train_ids, test_ids

    if no_val:
        return train, None , test
    else:
        return train, val, test  

def diversity_paderborn(
    df: pd.DataFrame,
    run: int,
    no_val: bool = False,
    splits: Literal["4x1", "3x2", "2x3", "1x4"] = "4x1",
    **kwargs
):
    """
    Split the bearings into train, validation and test sets

    Args:
        df (pd.DataFrame): dataframe with the metadata
        sampler (CombinationSampler): sampler instance
        run (int): run number
        healthy_bearing_ids (list): healthy bearing ids
        outer_bearing_ids (list): outer bearing ids
        inner_bearing_ids (list): inner bearing ids

    Returns:
        Tuple(list): train, val and test sets
    """
    if splits == "4x1":
        list1 = [
            ["K001"],
            ["K002"],
            ["K003"],
            ["K004"],
            ["K005"],
            ["K006"],
        ]
        list2 = [
            ["KA04"],
            ["KA15"],
            ["KA16"],
            ["KA22"],
            ["KA30"],
        ]
        list3= [
            ["KI04"],
            ["KI14"],
            ["KI16"],
            ["KI18"],
            ["KI21"],
            ["KI17"],
        ]
    elif splits == "3x2":
        list1 = [
            ["K001", "K002"],
            ["K001", "K003"],
            ["K001", "K004"],
            ["K001", "K005"],
            ["K001", "K006"],
            ["K002", "K003"],
            ["K002", "K004"],
            ["K002", "K005"],
            ["K002", "K006"],
            ["K003", "K004"],
            ["K003", "K005"],
            ["K003", "K006"],
            ["K004", "K005"],
            ["K004", "K006"],
            ["K005", "K006"],
        ]
        list2 = [
            ["KA04", "KA15"],
            ["KA04", "KA16"],
            ["KA04", "KA22"],
            ["KA04", "KA30"],
            ["KA15", "KA16"],
            ["KA15", "KA22"],
            ["KA15", "KA30"],
            ["KA16", "KA22"],
            ["KA16", "KA30"],
            ["KA22", "KA30"],
        ]
        list3 = [
            ["KI04", "KI14"],
            ["KI04", "KI16"],
            ["KI04", "KI18"],
            ["KI04", "KI21"],
            ["KI04", "KI17"],
            ["KI14", "KI16"],
            ["KI14", "KI18"],
            ["KI14", "KI21"],
            ["KI14", "KI17"],
            ["KI16", "KI18"],
            ["KI16", "KI21"],
            ["KI16", "KI17"],
            ["KI18", "KI21"],
            ["KI18", "KI17"],
            ["KI21", "KI17"],

        ]
    elif splits == "2x3":
        list1 = [
            ["K001", "K002", "K003"],
            ["K001", "K002", "K004"],
            ["K001", "K002", "K005"],
            ["K001", "K002", "K006"],
            ["K001", "K003", "K004"],
            ["K001", "K003", "K005"],
            ["K001", "K003", "K006"],
            ["K001", "K004", "K005"],
            ["K001", "K004", "K006"],
            ["K001", "K005", "K006"],
            ["K002", "K003", "K004"],
            ["K002", "K003", "K005"],
            ["K002", "K003", "K006"],
            ["K002", "K004", "K005"],
            ["K002", "K004", "K006"],
            ["K002", "K005", "K006"],
            ["K003", "K004", "K005"],
            ["K003", "K004", "K006"],
            ["K003", "K005", "K006"],
            ["K004", "K005", "K006"],
        ]
        list2 = [
            ["KA04","KA15","KA16"],
            ["KA04","KA15","KA22"],
            ["KA04","KA15","KA30"],
            ["KA04","KA16","KA22"],
            ["KA04","KA16","KA30"],
            ["KA04","KA22","KA30"],
            ["KA15","KA16","KA22"],
            ["KA15","KA16","KA30"],
            ["KA15","KA22","KA30"],
            ["KA16","KA22","KA30"],

        ]
        list3 = [
            ["KI04","KI14","KI16"],
            ["KI04","KI14","KI18"],
            ["KI04","KI14","KI21"],
            ["KI04","KI14","KI17"],
            ["KI04","KI16","KI18"],
            ["KI04","KI16","KI21"],
            ["KI04","KI16","KI17"],
            ["KI04","KI18","KI21"],
            ["KI04","KI18","KI17"],
            ["KI04","KI21","KI17"],
            ["KI14","KI16","KI18"],
            ["KI14","KI16","KI21"],
            ["KI14","KI16","KI17"],
            ["KI14","KI18","KI21"],
            ["KI14","KI18","KI17"],
            ["KI14","KI21","KI17"],
            ["KI16","KI18","KI21"],
            ["KI16","KI18","KI17"],
            ["KI16","KI21","KI17"],
            ["KI18","KI21","KI17"],
        ]
    elif splits == "1x4":
        list1 = [
            ['K001', 'K002', 'K003', 'K004']
            ,['K001', 'K002', 'K003', 'K005']
            ,['K001', 'K002', 'K003', 'K006']
            ,['K001', 'K002', 'K004', 'K005']
            ,['K001', 'K002', 'K004', 'K006']
            ,['K001', 'K002', 'K005', 'K006']
            ,['K001', 'K003', 'K004', 'K005']
            ,['K001', 'K003', 'K004', 'K006']
            ,['K001', 'K003', 'K005', 'K006']
            ,['K001', 'K004', 'K005', 'K006']
            ,['K002', 'K003', 'K004', 'K005']
            ,['K002', 'K003', 'K004', 'K006']
            ,['K002', 'K003', 'K005', 'K006']
            ,['K002', 'K004', 'K005', 'K006']
            ,['K003', 'K004', 'K005', 'K006']
                ]
        list2 = [
            ['KA04', 'KA15', 'KA16', 'KA22']
            ,['KA04', 'KA15', 'KA16', 'KA30']
            ,['KA04', 'KA15', 'KA22', 'KA30']
            ,['KA04', 'KA16', 'KA22', 'KA30']    
            ,['KA15', 'KA16', 'KA22', 'KA30']]
        list3 = [
            ['KI04', 'KI14', 'KI16', 'KI18']
            ,['KI04', 'KI14', 'KI16', 'KI21']
            ,['KI04', 'KI14', 'KI16', 'KI17']
            ,['KI04', 'KI14', 'KI18', 'KI21']
            ,['KI04', 'KI14', 'KI18', 'KI17']
            ,['KI04', 'KI14', 'KI21', 'KI17']
            ,['KI04', 'KI16', 'KI18', 'KI21']
            ,['KI04', 'KI16', 'KI18', 'KI17']
            ,['KI04', 'KI16', 'KI21', 'KI17']
            ,['KI04', 'KI18', 'KI21', 'KI17']
            ,['KI14', 'KI16', 'KI18', 'KI21']
            ,['KI14', 'KI16', 'KI18', 'KI17']
            ,['KI14', 'KI16', 'KI21', 'KI17']
            ,['KI14', 'KI18', 'KI21', 'KI17']
            ,['KI16', 'KI18', 'KI21', 'KI17']]

    sampler = CombinationSampler(list1, list2, list3, random_seed=42)
    combination = sampler.get_combination(run=run)

    test_ids = list(combination[0] + combination[1] + combination[2])
    remaining_healthy = list(set(HEALTHY_BEARING_IDS).difference(set(combination[0])))
    remaining_outer = list(set(OUTER_BEARING_IDS).difference(set(combination[1])))
    remaining_inner = list(set(INNER_BEARING_IDS).difference(set(combination[2])))

    random.seed(42)

    if splits == "4x1":
        # amostrate 4 IDs from each list of reamaining bearings to add to trainset (using random seed for reproducibility)
        remaining_healthy = random.sample(remaining_healthy, 4)
        remaining_outer = random.sample(remaining_outer, 4)
        remaining_inner = random.sample(remaining_inner, 4)
    elif splits == "3x2":
        remaining_healthy = random.sample(remaining_healthy, 3)
        remaining_outer = random.sample(remaining_outer, 3)
        remaining_inner = random.sample(remaining_inner, 3)
    elif splits == "2x3":
        remaining_healthy = random.sample(remaining_healthy, 2)
        remaining_outer = random.sample(remaining_outer, 2)
        remaining_inner = random.sample(remaining_inner, 2)
    elif splits == "1x4":
        remaining_healthy = random.sample(remaining_healthy, 1)
        remaining_outer = random.sample(remaining_outer, 1)
        remaining_inner = random.sample(remaining_inner, 1)

    train_ids = remaining_healthy + remaining_outer + remaining_inner

  
    train = df[df["bearing_id"].isin(train_ids)]
    test = df[df["bearing_id"].isin(test_ids)]

    val = test

    print(f"Test bearings: {test['bearing_id'].unique()}")
    print(f"Train bearings: {train['bearing_id'].unique()}")

    if no_val:
        return train, None , test
    else:
        return train, val, test  
  
def get_cv_splits(
    df: pd.DataFrame,
    run: int,
    healthy_bearing_ids: list,
    outer_bearing_ids: list,
    inner_bearing_ids: list,
    sampler: CombinationSampler=SAMPLER,
):

    combination = sampler.get_combination(run=run)

    train_ids = (
        list(set(healthy_bearing_ids).difference(set(combination[0])))
        + list(set(outer_bearing_ids).difference(set(combination[1])))
        + list(set(inner_bearing_ids).difference(set(combination[2])))
    )

    test_ids = list(combination[0] + combination[1] + combination[2])

    train_idx = df[df["bearing_id"].isin(train_ids)].index.values
    test_idx = df[df["bearing_id"].isin(test_ids)].index.values

    cv = (train_idx, test_idx)

    return cv

def paderborn_leakage(df, leakage_type: Literal["condition", "repetition", "segment", "no_leakage"], run: int = 0, **kwargs):
    """
    Split the DataFrame into training and test sets based on the specified leakage type.
    
    Parameters:
    - df: DataFrame containing the data.
    - leakage_type: Type of leakage to use for splitting.
    - run: Run number for the split.
    
    Returns:
    - train_df: Training DataFrame.
    - test_df: Test DataFrame.
    """

    rng = random.Random(run)

    def get_condition(row):
        if row['rotational_speed'] == 900:
            return 1
        elif row['load_torque'] == 0.1:
            return 2
        elif row['radial_force'] == 400:
            return 3
        else:
            return 0
    
    df['condition'] = df.apply(get_condition, axis=1)

    df.reset_index(drop=True, inplace=True)

    train_idx, test_idx = get_cv_splits(
        df=df,
        sampler=SAMPLER,
        run=run,
        healthy_bearing_ids=HEALTHY_BEARING_IDS,
        outer_bearing_ids=OUTER_BEARING_IDS,
        inner_bearing_ids=INNER_BEARING_IDS,
    )

    # Choose 1 condition to remove
    condition_to_remove = rng.choice([0, 1, 2, 3])

    # Choose 5 repetitions to remove
    repetitions_to_remove = rng.sample(range(0, 20), 5)

    train_df = df.iloc[train_idx]
    test_df = df.iloc[test_idx]

    print(f"Run {run}: Removing condition {condition_to_remove} and repetitions {repetitions_to_remove}")

    clean_train_idx = train_df[(train_df["condition"] != condition_to_remove) & (train_df["repetition"].astype(int).isin(repetitions_to_remove) == False)].index.values
    clean_test_idx = test_df[(test_df["condition"] != condition_to_remove) & (test_df["repetition"].astype(int).isin(repetitions_to_remove) == False)].index.values

    if leakage_type == "condition":
        test_idx = train_df[(train_df["condition"] == condition_to_remove) & (train_df["repetition"].astype(int).isin(repetitions_to_remove) == False)].index.values

    elif leakage_type == "repetition":
        test_idx = train_df[(train_df["condition"] != condition_to_remove) & (train_df["repetition"].astype(int).isin(repetitions_to_remove))].index.values

    elif leakage_type == "segment":
        test_idx = clean_train_idx
    elif leakage_type == "no_leakage":
        test_idx = clean_test_idx
    else:
        raise ValueError(f"Unknown leakage type: {leakage_type}. Choose from 'condition', 'repetition', 'segment', or 'no_leakage'.")

    train = df.loc[clean_train_idx].reset_index(drop=True)
    test = df.loc[test_idx].reset_index(drop=True)

    return train, None, test


# HUST

def hust_proposed_split(
    df, n_bearings_per_fault_mode=2, run=0, random_state=42, debug=False, no_val=False
):
    """
    Generate custom CV splits where each test set contains 2 IDs from each fault type.

    Parameters:
    - df: DataFrame with at least four columns representing class-wise IDs.
    - n_bearings_per_fault_mode: Number of bearings per fault mode in the test set.
    - run: Run number for reproducibility.
    - debug: Print debug info.

    Returns:
    - train, val, test: DataFrames for training, validation, and test sets.
    """

    # Get unique IDs per fault type
    inner_ids = ["I4", "I5", "I6", "I7", "I8"]
    outer_ids = ["O4", "O5", "O6", "O7", "O8"]
    ball_ids = ["B5", "B6", "B7", "B8"]
    inner_ball_ids = ["IB5", "IB6", "IB7", "IB8"]
    outer_ball_ids = ["OB4", "OB5", "OB6", "OB7", "OB8"]
    inner_outer_ids = ["IO4", "IO5", "IO6", "IO7", "IO8"]
    normal_ids = ["N4", "N5", "N6", "N7", "N8"]

    # All 2-combinations for each
    inner_combos = list(combinations(inner_ids, n_bearings_per_fault_mode))
    outer_combos = list(combinations(outer_ids, n_bearings_per_fault_mode))
    ball_combos = list(combinations(ball_ids, n_bearings_per_fault_mode))
    inner_ball_combos = list(combinations(inner_ball_ids, n_bearings_per_fault_mode))
    outer_ball_combos = list(combinations(outer_ball_ids, n_bearings_per_fault_mode))
    inner_outer_combos = list(combinations(inner_outer_ids, n_bearings_per_fault_mode))
    normal_combos = list(combinations(normal_ids, n_bearings_per_fault_mode))

    # Cartesian product to form full test sets (can be huge!)
    all_combos = list(
        product(
            inner_combos,
            outer_combos,
            ball_combos,
            inner_ball_combos,
            outer_ball_combos,
            inner_outer_combos,
            normal_combos,
        )
    )

    # Shuffle
    rng = random.Random(random_state)
    rng.shuffle(all_combos)

    # Limit to n_splits
    selected_combos = all_combos[run]
    print(f"Selected test IDs: {selected_combos}")

    (
        inner_test,
        outer_test,
        ball_test,
        inner_ball_test,
        outer_ball_test,
        inner_outer_test,
        normal_test,
    ) = selected_combos
    # Combine all test IDs into one set
    test_ids = (
        set(inner_test)
        | set(outer_test)
        | set(ball_test)
        | set(inner_ball_test)
        | set(outer_ball_test)
        | set(inner_outer_test)
        | set(normal_test)
    )
    test_mask = df["bearing_id"].isin(test_ids)

    test = df[test_mask]
    train = df[~test_mask]

    if debug:
        print(f"Inner Test IDs: {inner_test}")
        print(f"Outer Test IDs: {outer_test}")
        print(f"Ball  Test IDs: {ball_test}")
        print(f"Inner Ball Test IDs: {inner_ball_test}")
        print(f"Outer Ball Test IDs: {outer_ball_test}")
        print(f"Inner Outer Test IDs: {inner_outer_test}")
        print(f"Normal Test IDs: {normal_test}")
        print(f"Length of train: {len(train)}")
        print(f"Length of test: {len(test)}")

    val = test

    if no_val:
        return train, None, test
    return train, val, test
    
def hust_get_cv_splits_fixed(
    df,
    test_bearings_per_fault_mode=2,
    train_bearings_per_fault_mode=2,
    run=0,
    random_state=42,
    all_combined_on_test= False,
    no_combineds_on_train = False,
):
    """
    Generate custom CV splits where we control both training and test sets number of bearings. 
    """

   
    # Get unique IDs per fault type
    inner_ids = ["I4", "I5", "I6", "I7", "I8"]
    outer_ids = ["O4", "O5", "O6", "O7", "O8"]
    ball_ids = ["B5", "B6", "B7", "B8"]
    inner_ball_ids = ["IB5", "IB6", "IB7", "IB8"]
    outer_ball_ids = ["OB4", "OB5", "OB6", "OB7", "OB8"]
    inner_outer_ids = ["IO4", "IO5", "IO6", "IO7", "IO8"]
    normal_ids = ["N4", "N5", "N6", "N7", "N8"]

    all_combined = inner_ball_ids + outer_ball_ids + inner_outer_ids

    # All 2-combinations for each
    inner_combos = list(combinations(inner_ids, test_bearings_per_fault_mode))
    outer_combos = list(combinations(outer_ids, test_bearings_per_fault_mode))
    ball_combos = list(combinations(ball_ids, test_bearings_per_fault_mode))
    inner_ball_combos = list(combinations(inner_ball_ids, test_bearings_per_fault_mode))
    outer_ball_combos = list(combinations(outer_ball_ids, test_bearings_per_fault_mode))
    inner_outer_combos = list(combinations(inner_outer_ids, test_bearings_per_fault_mode))
    normal_combos = list(combinations(normal_ids, test_bearings_per_fault_mode))

    # Cartesian product to form full test sets (can be huge!)

    all_combos = list(
        product(
            inner_combos,
            outer_combos,
            ball_combos,
            inner_ball_combos,
            outer_ball_combos,
            inner_outer_combos,
            normal_combos,
        )
    )

    # Shuffle
    rng = random.Random(random_state)
    rng.shuffle(all_combos)

    # Limit to n_splits
    selected_test_combos = all_combos[run]

    (
        inner_test,
        outer_test,
        ball_test,
        inner_ball_test,
        outer_ball_test,
        inner_outer_test,
        normal_test,
    ) = selected_test_combos

    

    remaining_inner = set(inner_ids) - set(inner_test)
    remaining_outer = set(outer_ids) - set(outer_test)
    remaining_ball = set(ball_ids) - set(ball_test)
    remaining_inner_ball = set(inner_ball_ids) - set(inner_ball_test)
    remaining_outer_ball = set(outer_ball_ids) - set(outer_ball_test)
    remaining_inner_outer = set(inner_outer_ids) - set(inner_outer_test)
    remaining_normal = set(normal_ids) - set(normal_test)

    inner_combos = list(combinations(remaining_inner, train_bearings_per_fault_mode))
    outer_combos = list(combinations(remaining_outer, train_bearings_per_fault_mode))
    ball_combos = list(combinations(remaining_ball, train_bearings_per_fault_mode))
    inner_ball_combos = list(combinations(remaining_inner_ball, train_bearings_per_fault_mode))
    outer_ball_combos = list(combinations(remaining_outer_ball, train_bearings_per_fault_mode))
    inner_outer_combos = list(combinations(remaining_inner_outer, train_bearings_per_fault_mode))
    normal_combos = list(combinations(remaining_normal, train_bearings_per_fault_mode))

    all_combos = list(
        product(
            inner_combos,
            outer_combos,
            ball_combos,
            inner_ball_combos,
            outer_ball_combos,
            inner_outer_combos,
            normal_combos,
        )
    )

    rng.shuffle(all_combos)

    if run >= len(all_combos):
        selected_combos = all_combos[-1] # Use last combination if run exceeds number of combinations. This will hapen when training_bearings=1
    else:
        selected_combos = all_combos[run]

    (
        inner_train,
        outer_train,
        ball_train,
        inner_ball_train,
        outer_ball_train,
        inner_outer_train,
        normal_train,
    ) = selected_combos
    
    print(f"Selected train IDs: {selected_combos}")

    train_ids = (
        set(inner_train)
        | set(outer_train)
        | set(ball_train)
        | set(inner_ball_train)
        | set(outer_ball_train)
        | set(inner_outer_train)
        | set(normal_train)
    )

    test_ids = (
        set(inner_test)
        | set(outer_test)
        | set(ball_test)
        | set(inner_ball_test)
        | set(outer_ball_test)
        | set(inner_outer_test)
        | set(normal_test)
    )

    if no_combineds_on_train:
        # Remove any id in all_combined from train_ids
        train_ids = train_ids - set(all_combined)

    if all_combined_on_test:
        test_ids = test_ids | set(all_combined)
        # Remove any id in all_combined from train_ids
        train_ids = train_ids - set(all_combined)

        
    print(f"Final train IDs: {train_ids}")
    print(f"Final test IDs: {test_ids}")

    train_mask = df["bearing_id"].isin(train_ids)
    test_mask = df["bearing_id"].isin(test_ids)
    test_idx = df[test_mask].index.values.tolist()
    train_idx = df[train_mask].index.values.tolist()

    cv = (train_idx, test_idx)

    return cv

def hust_fixed_split(df, run=0, random_state=42, test_bearings_per_fault_mode=2, train_bearings_per_fault_mode=2, no_val=False):

    print(f"Generating split with {train_bearings_per_fault_mode} bearings per fault mode in train and {test_bearings_per_fault_mode} bearings per fault mode in test for run {run} with random state {random_state}")

    train, test =  get_cv_splits_fixed(df, test_bearings_per_fault_mode=test_bearings_per_fault_mode, train_bearings_per_fault_mode=train_bearings_per_fault_mode, run=run, random_state=random_state)
    
    train = df.iloc[train]
    test = df.iloc[test]

    if no_val:
        return train, None, test

    else:
        val = test
        return train, val, test
