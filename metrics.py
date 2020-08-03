import numpy as np


def nrmse(y_hat, y):
    """
        Normalized root mean squared error
    """
    meaned_sqd_diff = np.mean(np.power(y_hat - y, 2))
    return np.sqrt(meaned_sqd_diff) / np.std(y)
