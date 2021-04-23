import os
import re
import math
import warnings
import argparse
import pathlib as pb
from pathlib import Path  # TODO: Fix this KOSTYL due to merge of different utils files

import yaml
import numpy as np
import tensorly as tl

warnings.filterwarnings("ignore", category=RuntimeWarning) 



def tensorify(X, y, shape):
    tensor = np.full(shape, np.nan)

    for i, d in enumerate(X):
        lat, lon, t = d.astype(np.int)
        tensor[lat, lon, t] = y[i]

    return tensor


def center_mat(mat):
    nan_mask = np.isnan(mat)
    temp_mat = mat.copy()
    temp_mat[nan_mask] = 0

    m0 = temp_mat.mean(axis=0)
    for i in range(temp_mat.shape[0]):
        temp_mat[i, :] -= m0

    m1 = temp_mat.mean(axis=1)
    for i in range(temp_mat.shape[1]):
        temp_mat[:, i] -= m1

    temp_mat[nan_mask] = np.nan
    return temp_mat, m0, m1


def decenter_mat(mat, m0, m1):
    temp_mat = mat.copy()

    for i in range(temp_mat.shape[0]):
        temp_mat[i, :] += m0

    for i in range(temp_mat.shape[1]):
        temp_mat[:, i] += m1

    return temp_mat


def center_3d_tensor(tensor, lat_lon_separately=True):
    temp_tensor = tensor.copy()

    # Spatial centering
    spatial_means = []
    if lat_lon_separately:
        m0 = np.nanmean(temp_tensor, axis=0)
        m0[np.isnan(m0)] = 0
        for i in range(temp_tensor.shape[0]):
            temp_tensor[i, :, :] -= m0
        spatial_means.append(m0)

        m1 = np.nanmean(temp_tensor, axis=1)
        m1[np.isnan(m1)] = 0
        for i in range(temp_tensor.shape[1]):
            temp_tensor[:, i, :] -= m1
        spatial_means.append(m1)
    else:
        # For each day unified spatial mean (in flattened spatial array)
        for t in range(temp_tensor.shape[-1]):
            mat_mean = np.nanmean(temp_tensor[:, :, t])
            if np.isnan(mat_mean):
                mat_mean = 0
            temp_tensor[:, :, t] -= mat_mean
            spatial_means.append(mat_mean)

    # Time centering
    m2 = np.nanmean(temp_tensor, axis=2)
    m2[np.isnan(m2)] = 0
    for i in range(temp_tensor.shape[2]):
        temp_tensor[:, :, i] -= m2

    return temp_tensor, spatial_means, m2


def decenter_3d_tensor(tensor, spatial_means, m2, lat_lon_separately=True):
    temp_tensor = tensor.copy()

    # Spatial decentering
    if lat_lon_separately:
        m0, m1 = spatial_means
        for i in range(temp_tensor.shape[0]):
            temp_tensor[i, :, :] += m0
        for i in range(temp_tensor.shape[1]):
            temp_tensor[:, i, :] += m1
    else:
        for t in range(temp_tensor.shape[-1]):
            temp_tensor[:, :, t] += spatial_means[t]

    # Time decentering
    for i in range(temp_tensor.shape[2]):
        temp_tensor[:, :, i] += m2

    return temp_tensor


def rectify_tensor(tensor):
    rect_mat = []
    for t in range(tensor.shape[-1]):
        rect_mat.append(tensor[:, :, t].flatten())
    rect_mat = np.array(rect_mat)
    rect_mat = np.moveaxis(rect_mat, 0, -1)
    return rect_mat


def unrectify_mat(mat, spatial_shape):
    tensor = []

    for t in range(mat.shape[-1]):
        col = mat[:, t]
        unrectified_col = col.reshape(spatial_shape)
        tensor.append(unrectified_col)

    tensor = np.array(tensor)
    tensor = np.moveaxis(tensor, 0, -1)

    return tensor


def nrmse(y_hat, y):
    """
        Normalized root mean squared error
    """
    root_meaned_sqd_diff = np.sqrt(np.mean(np.power(y_hat - y, 2)))
    return root_meaned_sqd_diff / np.std(y)


def calculate_mat_energy(mat, s):
    sample_count_0 = mat.shape[1]
    sample_coef_0 = 1 / (sample_count_0 - 1)
    total_energy_0 = np.array([sample_coef_0 * np.trace(mat @ mat.T) for _ in range(len(s))])
    expl_energy_0 = -np.sort(-sample_coef_0 * s * s)
    expl_energy_ratio_0 = expl_energy_0 / total_energy_0

    sample_count_1 = mat.shape[0]
    sample_coef_1 = 1 / (sample_count_1 - 1)
    total_energy_1 = np.array([sample_coef_1 * np.trace(mat.T @ mat) for _ in range(len(s))])
    expl_energy_1 = -np.sort(-sample_coef_1 * s * s)
    expl_energy_ratio_1 = expl_energy_1 / total_energy_1

    return np.array([[total_energy_0, expl_energy_0, expl_energy_ratio_0],
                        [total_energy_1, expl_energy_1, expl_energy_ratio_1]])

def calculate_tucker_energy(tensor, A):
    energy_stack = []
    for i in range(tensor.ndim):
        sample_count_i = np.prod(tensor.shape) / tensor.shape[i]
        sample_coef_i = 1 / (sample_count_i - 1)
        unfold_i = tl.unfold(tensor, i)
        tensor_proj_i = tl.tenalg.mode_dot(tensor, A[i].T, i)
        tensor_proj_unfold_i = tl.unfold(tensor_proj_i, i)

        full_cov_i = sample_coef_i * unfold_i @ unfold_i.T
        tensor_proj_cov_i = sample_coef_i * tensor_proj_unfold_i @ tensor_proj_unfold_i.T

        total_energy_i = np.trace(full_cov_i)
        expl_energy_i_per_component = -np.sort(-tensor_proj_cov_i.diagonal())
        expl_energy_ratio_i_per_component = expl_energy_i_per_component / total_energy_i
        total_energy_i_per_component = [total_energy_i for _ in range(len(expl_energy_i_per_component))]

        energy_stack.append([total_energy_i_per_component,
                                expl_energy_i_per_component,
                                expl_energy_ratio_i_per_component])
    energy_stack = np.array(energy_stack)

    return energy_stack


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
