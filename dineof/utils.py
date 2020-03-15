import math
import os

import numpy as np


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
