"""
Scaling functions for the Pytorch Custom Dataset.
"""

import numpy as np
import torch.nn as nn
import torch


def combine(averages, variances, counts):
    """
    Combine averages and variances to one single average and variance.

    Parameters
    ----------
        averages: 2d numpy array of averages for each part. (num_parts, num_features)
        variances: 2d numpy array of variances for each part. (num_parts, num_features)
        counts: List of number of elements in each part.
    Returns
    -------
        average: Average for each feature over all parts.
        variance: Variance for each feature over all parts.
    """
    size = np.sum(counts)
    average = []
    variance = []
    for column in range(0, averages.shape[1]):
        average_column = averages[:, column]
        variance_column = variances[:, column]

        total_average_column = np.average(average_column, weights=counts)
        squares = (counts - 1) * variance_column + counts * (
            average_column - total_average_column
        ) ** 2
        total_variance_column = np.sum(squares) / (size - 1)

        average.append(total_average_column)
        variance.append(total_variance_column)

    return np.array(average), np.array(variance)


def std_scaling(arr: np.array, **params) -> np.array:
    """
    Standard scaling implementation to be used with our Custom Dataset.

    Parameters
    ----------
    arr : np.array
        Any 1D, 2D signal input.
    **params : dict
        Dictionary with scaling parameters.

    Returns
    -------
    np.array
        Scaled output.
    """

    if params["norm_type"] == "dataset-wise":
        scaled_arr = (arr - params["mean"]) / params["std"]
    if params["norm_type"] == "entry-wise":
        scaled_arr = (arr - np.mean(arr)) / np.std(arr)
    return scaled_arr


def max_scaling(arr: np.array, **params) -> np.array:
    """
    Max scaling implementation to be used with our Custom Dataset.

    Parameters
    ----------
    arr : np.array
        Any 1D, 2D signal input.
    **params : dict
        Dictionary with scaling parameters.

    Returns
    -------
    np.array
        Scaled output.
    """

    if params["norm_type"] == "dataset-wise":
        scaled_arr = arr / params["max"]
    if params["norm_type"] == "entry-wise":
        scaled_arr = arr / np.max(arr)
    return scaled_arr


def normalization_factor(arr: np.array, **params) -> np.array:
    """
    Normalization factor implementation to be used with our Custom Dataset.


    Parameters
    ----------
    arr : np.array
        Any 1D, 2D signal input.
    **params : dict
        Dictionary with scaling parameters

    Returns
    -------
    np.array
        Scaled output.
    """

    scaled_arr = arr / params["factor"]

    return scaled_arr


class MinMaxScaler(nn.Module):
    """
    Min-max scaling implementation to be used with our Custom Dataset.

    Parameters
    ----------
    arr : np.array
        Any 1D, 2D signal input.
    Returns
    -------
    np.array
        Scaled output.
    """

    def __init__(self):
        super().__init__()

    def __call__(self, arr: torch.Tensor) -> torch.Tensor:
        scaled_arr = (arr - torch.min(arr)) / (torch.max(arr) - torch.min(arr))
        return scaled_arr
