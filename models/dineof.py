import os
import sys
import numpy as np
from scipy.sparse.linalg import svds
from sklearn.base import BaseEstimator
from tqdm import tqdm
file_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(file_dir))
import tensor_utils as tu
import metrics


class DINEOF(BaseEstimator):
    def __init__(self, K, tensor_shape,
                 nitemax=300, toliter=1e-3, tol=1e-8, to_center=True, 
                 keep_non_negative_only=True,
                 with_energy=True):
        self.K = K
        self.nitemax = nitemax
        self.toliter = toliter
        self.tol = tol
        self.to_center = to_center
        self.keep_non_negative_only = keep_non_negative_only
        self.tensor_shape = tensor_shape
        self.with_energy = with_energy
        
    def score(self, X, y):
        """
            You can think of this like negative error (bigger is better due to error diminishing.
            It is made like so to be compatible with scikit-learn grid search utilities.
        """
        y_hat = self.predict(X)
        return -metrics.nrmse(y_hat, y)
        
    def predict(self, X):
        output = np.array([self.reconstructed_tensor[x[0], x[1], x[2]] for x in X])
        return output
        
    def fit(self, X, y):
        # X - is a spreadshit in format (lat, lon, t) here
        # Target of this func to prepare input for _fit
        tensor = np.full(self.tensor_shape, np.nan)
        for i, x in enumerate(X):
            lat, lon, t = x
            tensor[lat, lon, t] = y[i]
        
        self._fit(tu.rectify_3d_tensor(tensor))
        
    def _fit(self, mat):
        if self.to_center:
            mat, *means = tu.center_mat(mat)

        # Initial guess
        nan_mask = np.isnan(mat)
        mat[nan_mask] = 0

        conv_error = 0
        energy_per_iter = []
        for i in tqdm(range(self.nitemax)):
            u, s, vt = svds(mat, k=self.K, tol=self.tol)

            # Save energy characteristics for this iteration
            if self.with_energy:
                energy_i = tu.calculate_mat_energy(mat, s)
                energy_per_iter.append(energy_i)
            
            mat_hat = u @ np.diag(s) @ vt
            mat_hat[~nan_mask] = mat[~nan_mask]
            diff_in_clouds = mat_hat[nan_mask] - mat[nan_mask]
            new_conv_error = np.linalg.norm(diff_in_clouds) / np.std(mat[~nan_mask])
            mat = mat_hat
            if (new_conv_error <= self.toliter) or (abs(new_conv_error - conv_error) < self.toliter):
                break
            conv_error = new_conv_error

        energy_per_iter = np.array(energy_per_iter)

        if self.to_center:
            mat = tu.decenter_mat(mat, *means)

        if self.keep_non_negative_only:
            mat[mat < 0] = 0

        # Save energies in model for distinct components (lat, lon, t)
        if self.with_energy:
            for i in range(mat.ndim):
                setattr(self, f'total_energy_{i}', np.array(energy_per_iter[:, i, 0]))
                setattr(self, f'explained_energy_{i}', np.array(energy_per_iter[:, i, 1]))
                setattr(self, f'explained_energy_ratio_{i}', np.array(energy_per_iter[:, i, 2]))

        self.final_iter = i
        self.conv_error = new_conv_error
        self.reconstructed_tensor = tu.unrectify_mat(mat, spatial_shape=self.tensor_shape[:-1])
        self.singular_values_ = s
        self.ucomponents_ = u
        self.vtcomponents_ = vt
