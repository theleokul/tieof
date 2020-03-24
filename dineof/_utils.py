import math
import os

import numpy as np
from pathlib import Path


def guard(log_expr, error):
    if not log_expr:
        raise Exception(error)


def floor_float(n):
    """Find the most significant digit in float and floor to it"""
    if n == 0:
        return 0

    sgn = -1 if n < 0 else 1
    scale = int(-math.floor(math.log10(abs(n))))

    if scale <= 0:
        scale = 1
    factor = 10 ** scale

    return sgn * math.floor(abs(n) * factor) / factor


def ls(dir_path, root_replacer=None):
    res = []

    root, _, filenames = next(os.walk(dir_path))
    for filename in filenames:
        path = os.path.join(root_replacer, filename) if root_replacer else os.path.join(root, filename)
        res.append(path)

    return sorted(res)


def calculate_fullness(X, mask):
    """
    Calculate fullness F (0<=F<=1)

    X - 2D matrix with NaN for missing points
    Mask - 2D matrix with 1 - lake, 0 - land
    """
    X = X[mask.astype(np.bool)]
    data_size = np.count_nonzero(~np.isnan(X))
    overall_size = X.size

    return data_size / overall_size


def zero_negative(X):
    not_nan_mask = ~np.isnan(X)
    X[not_nan_mask] = np.clip(X[not_nan_mask], 0, X[not_nan_mask].max())
    return X


def apply_log_scale(X, small_chunk_to_add=0):
    not_nan_mask = ~np.isnan(X)
    X[not_nan_mask] = np.log(X[not_nan_mask] + small_chunk_to_add)
    return X


def get_min(X):
    m = X.min()
    if np.isnan(m):
        m = X[~np.isnan(X)].min()
    return m


def get_max(X):
    m = X.max()
    if np.isnan(m):
        m = X[~np.isnan(X)].max()
    return m


def get_mean(X, apply_log_scale=False):
    m = X.mean()
    if np.isnan(m):
        m = X[~np.isnan(X)].mean()
    if apply_log_scale:
        m = np.log(m)
    return m


def get_std(X, apply_log_scale=False):
    std = X.std()
    if np.isnan(std):
        std = X[~np.isnan(X)].std()
    if apply_log_scale:
        std = np.log(std)
    return std


def remove_extension(path):
    return os.path.join(os.path.dirname(path), Path(path).stem)


def get_matrix_by_day(day, day_mapper, tensor):
    """Get matrix by day, if day is not found => return empty matrix (filled with nans)"""
    try:
        day_index = np.where(day_mapper == day)[0][0]
    except IndexError:
        return np.full_like(tensor[:, :, 0], np.nan)
    return tensor[:, :, day_index]


def form_tensor(X, multiplicator, move_new_axis_to_end=True):
    tensor = []
    for _ in range(multiplicator):
        tensor.append(X)
    tensor = np.array(tensor)

    if move_new_axis_to_end:
        tensor = np.moveaxis(tensor, 0, -1)

    return tensor
