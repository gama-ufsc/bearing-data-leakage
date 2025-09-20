import random
from itertools import combinations, product
from typing import Literal, Tuple
from sklearn.model_selection import GroupShuffleSplit

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# ------------------ Global variables  ------------------

# Real fault bearings used in the paper experiments
healthy_bearing_ids = ["K001", "K002", "K003", "K004", "K005"]
outer_bearing_ids = ["KA04", "KA15", "KA16", "KA22", "KA30"]
inner_bearing_ids = ["KI04", "KI14", "KI16", "KI18", "KI21"]

# Remaining real fault and normal condition bearings
remaining_healthy_bearings_ids = ["K006"]
remaining_outer_bearings_ids = ["KI17"]
combination_bearings_ids = ["KB23", "KB24", "KB27"]

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

# Following the rolling bearing table from the paper
# we have 3 columns: healthy, outer, inner
bearing_table = np.array(
    [
        [healthy, outer, inner]
        for healthy, outer, inner in zip(
            healthy_bearing_ids, outer_bearing_ids, inner_bearing_ids
        )
    ]
)

# Fault bearing operating conditions
# each condition is a row, where the columns are:
# [rotational_speed, load_torque, radial_force]
operating_conditions = np.array(
    [[1500, 0.7, 1000], [900, 0.7, 1000], [1500, 0.1, 1000], [1500, 0.7, 400]]
)

# ------------------ Leakage Paderborn data splits ------------------


def train_test_split_leakage_1(
    df: pd.DataFrame,
    row: int,
    condition: int,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple:
    """
    Choose one table row and one of the 4 operating conditions
    Ex: row 0 = ["K001", "KA04", "KI04"] and condition 0 = [1500, 0.7, 1000]

    Args:
        df (pd.DataFrame): dataframe with the metadata
        row (int): row number of the bearing table
        condition (int): operating condition number
        test_size (float): test size
        random_state (int): random state

    Returns:
        Tuple(pd.DataFrame): train, val and test dataframes
    """
    split = df[
        (df["bearing_id"].isin(bearing_table[row]))
        & (df["rotational_speed"] == operating_conditions[condition][0])
        & (df["load_torque"] == operating_conditions[condition][1])
        & (df["radial_force"] == operating_conditions[condition][2])
    ]

    train, test = train_test_split(
        split, test_size=test_size, random_state=random_state
    )
    val = test

    return train, val, test


def train_test_split_leakage_2(
    df: pd.DataFrame,
    row: int,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple:
    """
    Choose one table row and use all operating conditions (80 signals for each label)

    Args:
        df (pd.DataFrame): dataframe with the metadata
        row (int): row number of the bearing table
        test_size (float): test size
        random_state (int): random state

    Returns:
        Tuple(pd.DataFrame): train, val and test dataframes
    """
    split = df[(df["bearing_id"].isin(bearing_table[row]))]

    train, test = train_test_split(
        split, test_size=test_size, random_state=random_state
    )
    val = test

    return train, val, test


def ottawa_proposed_split(
    df,
    n_bearings_per_fault_mode=2,
    run=0,
    random_state=42,
    debug=False,
    return_ids: bool = False,
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

    val = test

    if not return_ids:
        return train, val, test
    else:
        return inner_test, outer_test, ball_test


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


def cwru_old_optimization_split(
    df: pd.DataFrame,
    test_fold: Literal[1, 2, 3],
    withHP0: bool = False,
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

    if (train_path is not None) and (test_path is not None):
        if not withHP0:
            train_path = f"data/splits/cwru/fold_{test_fold}/train_oldapproach.csv"
            test_path = f"data/splits/cwru/fold_{test_fold}/test_oldapproach.csv"
        else:
            train_path = (
                f"data/splits/cwru/fold_{test_fold}/train_oldapproach_withHP0.csv"
            )
            test_path = (
                f"data/splits/cwru/fold_{test_fold}/test_oldapproach_withHP0.csv"
            )
    train_ids = pd.read_csv(train_path)["waveform_id"].tolist()
    test_ids = pd.read_csv(test_path)["waveform_id"].tolist()

    train = df[df["waveform_id"].isin(train_ids)].reset_index(drop=True)
    test = df[df["waveform_id"].isin(test_ids)].reset_index(drop=True)

    val = test

    return train, val, test


def split_cwru_proposed(
    df: pd.DataFrame,
    normal_train_side: Literal["DE", "FE"],
    random_state: int = 42,
    HP0_ontest: bool = False,
    HP0_ontrain: bool = False,
    include_normal_config_on_test: bool = False,
    test_size: float = 1 / 3,
) -> Tuple[pd.DataFrame, None, pd.DataFrame]:
    """Prepare CWRU dataset for model training and testing.

    - LT Split (Fault location and type): stratified, (Size): random
    """
    df_train = pd.DataFrame()
    df_test = pd.DataFrame()

    df["fault_type"] = df["fault_type"].str.replace("OR@3", "OR")
    df["fault_type"] = df["fault_type"].str.replace("OR@6", "OR")
    df["group"] = (
        df["fault_location"].astype(str)
        + "_"
        + df["fault_type"].astype(str)
        + "_"
        + df["fault_size"].astype(str)
    )

    # We will not use all 411 signals from CWRU, this CSV filter to only used (144 signals)
    df_filtered_ids = pd.read_csv("data/splits/cwru/filtered_cwru.csv")
    df = df[df["waveform_id"].isin(df_filtered_ids["waveform_id"].tolist())]

    test_ids_optimization = []

    for optimization_fold in range(1, 4):
        for train_side in ["DE", "FE"]:
            test_ids = pd.read_csv(
                f"data/splits/cwru/fold_{optimization_fold}/test_{train_side}.csv"
            )["waveform_id"].tolist()
            test_ids_optimization.append(test_ids)

    # For each location-type pair, split into train and test by randomly selecting a size
    for enum, (_group, df_group) in enumerate(df.groupby(["fault_location", "fault_type"])):
        gss = GroupShuffleSplit(
            n_splits=1, test_size=test_size, random_state=random_state + enum * 1000
        )
        df_train_idx, df_test_idx = next(gss.split(df_group, groups=df_group.group))
        df_train = pd.concat([df_train, df_group.iloc[df_train_idx]], ignore_index=True)
        df_test = pd.concat([df_test, df_group.iloc[df_test_idx]], ignore_index=True)

        print(df_train.shape, df_test.shape)

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

    # Check if the test set is equal to some of the optimization test sets and if so, create a new test set
    test_waveform_ids = set(df_test["waveform_id"].tolist())

    # Function to check if test set is equal to any optimization set
    def check_equality(test_ids, opt_ids_list):
        for i, opt_ids in enumerate(opt_ids_list):
            opt_ids_set = set(opt_ids)
            if test_ids == opt_ids_set:
                return True
        return False

    # Initial check before any additional splits
    is_equal = check_equality(test_waveform_ids, test_ids_optimization)
    attempt = 0

    while is_equal:
        attempt += 1
        new_random_state = random_state + attempt * 100

        # Repeat the splitting process with a new random seed
        df_train = pd.DataFrame()
        df_test = pd.DataFrame()

        for enum, (_group, df_group) in enumerate(df.groupby(["fault_location", "fault_type"])):
            gss = GroupShuffleSplit(
                n_splits=1, test_size=test_size, random_state=new_random_state + enum * 1000
            )
            df_train_idx, df_test_idx = next(gss.split(df_group, groups=df_group.group))
            df_train = pd.concat(
                [df_train, df_group.iloc[df_train_idx]], ignore_index=True
            )
            df_test = pd.concat(
                [df_test, df_group.iloc[df_test_idx]], ignore_index=True
            )

        # Re-apply the filtering based on normal_train_side
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

        # Check equality again
        test_waveform_ids = set(df_test["waveform_id"].tolist())
        is_equal = check_equality(test_waveform_ids, test_ids_optimization)

    #df_train = df_train.drop(columns=["group"])
    #df_test = df_test.drop(columns=["group"])

    if not HP0_ontest:
        print("Removing HP0 from test set")
        df_test = df_test[~df_test["load"].isin([0])]
    else:
        print("Keeping HP0 in test set")
    if not HP0_ontrain:
        print("Removing HP0 from train set")
        df_train = df_train[~df_train["load"].isin([0])]
    else:
        print("Keeping HP0 in train set")

    if include_normal_config_on_test:
        normal_df = pd.read_csv("data/splits/cwru/normal_signals.csv")
        df_test = pd.concat([df_test, normal_df], ignore_index=True)
    print("Test set size: ", df_test.shape[0])

    df_val = df_test
    return df_train, df_val, df_test

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
    save_combination: bool = False,
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
    if save_combination:
        pd.Series([test_ids]).to_pickle(
            f"data/training_combinations/test_ids_run_{run}.pkl",
        )

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

    if no_val:
        return train, None , test
    else:
        return train, val, test
