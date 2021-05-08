import os
import sys

from tqdm import trange
from loguru import logger
import numpy as np
from scipy.sparse.linalg import svds
from sklearn.base import BaseEstimator

file_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(file_dir)
import model_utils as utils



class DINEOF(BaseEstimator):
    def __init__(self, R, tensor_shape, mask=None,
                 nitemax=300, toliter=1e-3, tol=1e-8, to_center=True, 
                 keep_non_negative_only=True,
                 with_energy=False,
                 early_stopping=True):
        self.K = R  # HACK: Make interface consistent with DINEOF3, but want to keep intrinsics as is
        self.nitemax = nitemax
        self.toliter = toliter
        self.tol = tol
        self.to_center = to_center
        self.keep_non_negative_only = keep_non_negative_only
        self.tensor_shape = tensor_shape
        self.with_energy = with_energy
        self.mask = np.load(mask).astype(bool) if mask is not None else np.ones(tensor_shape, type=bool)
        self.mask = self._broadcast_mask(self.mask, tensor_shape[-1])
        self.inverse_mask = ~self.mask
        self.early_stopping = early_stopping

    def _broadcast_mask(self, mask, t):
        mask = np.repeat(mask[:, :, None], t, axis=2)
        return utils.rectify_tensor(mask)
        
    def score(self, X, y):
        """
            You can think of this like negative error (bigger is better due to error diminishing.
            It is made like so to be compatible with scikit-learn grid search utilities.
        """
        y_hat = self.predict(X)
        return -utils.nrmse(y_hat, y)
    
    def rmse(self, X, y):
        return -self.score(X, y) * y.std()
    
    def nrmse(self, X, y):
        return -self.score(X, y)
        
    def predict(self, X):
        output = np.array([self.reconstructed_tensor[x[0], x[1], x[2]] for x in X])
        return output
        
    def fit(self, X, y):
        tensor = utils.tensorify(X, y, self.tensor_shape)
        self._fit(utils.rectify_tensor(tensor))
        
    def _fit(self, mat):
        if mat.ndim > 2:
            mat = utils.rectify_tensor(mat)

        if self.to_center:
            mat, *means = utils.center_mat(mat)

        # Initial guess
        nan_mask = np.logical_and(np.isnan(mat), self.mask)
        non_nan_mask = np.logical_and(~nan_mask, self.mask)
        mat[nan_mask] = 0
        # Outside of an investigated area everything is considered to be zero
        mat[self.inverse_mask] = 0 

        pbar = trange(self.nitemax, desc='Reconstruction')
        conv_error = 0
        energy_per_iter = []
        for i in pbar:
            u, s, vt = svds(mat, k=self.K, tol=self.tol)

            # Save energy characteristics for this iteration
            if self.with_energy:
                energy_i = utils.calculate_mat_energy(mat, s)
                energy_per_iter.append(energy_i)
            
            mat_hat = u @ np.diag(s) @ vt
            mat_hat[non_nan_mask] = mat[non_nan_mask]
            mat_hat[self.inverse_mask] = 0

            new_conv_error = np.sqrt(np.mean(np.power(mat_hat[nan_mask] - mat[nan_mask], 2))) / mat[non_nan_mask].std()
            mat = mat_hat

            pbar.set_postfix(error=new_conv_error, rel_error=abs(new_conv_error - conv_error))
            
            grad_conv_error = abs(new_conv_error - conv_error)
            conv_error = new_conv_error
            
            logger.info(f'Error/Relative Error at iteraion {i}: {conv_error}, {grad_conv_error}')
            
            if self.early_stopping:
                break_condition = (conv_error <= self.toliter) or (grad_conv_error < self.toliter)
            else:
                break_condition = (conv_error <= self.toliter)
                
            if break_condition:              
                break

        energy_per_iter = np.array(energy_per_iter)

        if self.to_center:
            mat = utils.decenter_mat(mat, *means)

        if self.keep_non_negative_only:
            mat[mat < 0] = 0

        mat[self.inverse_mask] = np.nan

        # Save energies in model for distinct components (lat, lon, t)
        if self.with_energy:
            for i in range(mat.ndim):
                setattr(self, f'total_energy_{i}', np.array(energy_per_iter[:, i, 0]))
                setattr(self, f'explained_energy_{i}', np.array(energy_per_iter[:, i, 1]))
                setattr(self, f'explained_energy_ratio_{i}', np.array(energy_per_iter[:, i, 2]))

        self.final_iter = i
        self.conv_error = conv_error
        self.grad_conv_error = grad_conv_error
        self.reconstructed_tensor = utils.unrectify_mat(mat, spatial_shape=self.tensor_shape[:-1])
        self.singular_values_ = s
        self.ucomponents_ = u
        self.vtcomponents_ = vt
