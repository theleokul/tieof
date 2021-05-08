import os
import sys
import subprocess
import tempfile
import argparse as ap
import pathlib as pb

import yaml
import numpy as np
from loguru import logger

ROOT_PATH = pb.Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_PATH))
import model.model_utils as mutils
import interpolator._interpolator as I



class DINEOFGHER:

    def __init__(self, config: ap.Namespace):
        # Basically config consists of arguments for gher dineof cli
        for k, v in vars(config).items():
            setattr(self, k, v)

    def fit(
        self

        , unified_tensor_path  # Correct order of axes: (lat, lon, t)
        , mask_path
        , timeline_path
        
        , output_dir
        , output_stem

        , zero_negative_in_result_tensor=True
    ):

        os.makedirs(output_dir, exist_ok=True)

        self.unified_tensor_dat_path = self._to_dat(unified_tensor_path)
        self.mask_dat_path = self._to_dat(mask_path)
        self.timeline_dat_path = self._to_dat(timeline_path)

        self.unified_tensor_npy_path = self._to_npy(unified_tensor_path)
        self.mask_npy_path = self._to_npy(mask_path)
        self.timeline_npy_path = self._to_npy(timeline_path)

        self.result_tensor_dat_path = os.path.join(output_dir, output_stem + '.dat')
        self.result_tensor_npy_path = os.path.join(output_dir, output_stem + '.npy')

        with tempfile.NamedTemporaryFile(suffix='.init') as tmp:
            tmp.write(self._construct_dineof_init(
                data=self.unified_tensor_dat_path
                , mask=self.mask_dat_path
                , time=self.timeline_dat_path
                , output_dir=output_dir
                , result_tensor_dat_path=self.result_tensor_dat_path
            ))
            tmp.seek(0)
            completed_process = subprocess.run([
                f'{self.dineof_executer}',
                f'{tmp.name}'
            ], stdout=subprocess.PIPE)
            output = completed_process.stdout
            logger.info(output)
            self.stats = self.parse_output(output)
            logger.info(str(self.stats))

        # Save output of GHER DINEOF in .npy format
        I.Interpolator.dat_to_npy(self.result_tensor_dat_path, self.result_tensor_npy_path)

        if zero_negative_in_result_tensor:
            result_tensor = np.load(self.result_tensor_npy_path)
            result_tensor = mutils.zero_negative(result_tensor)
            np.save(self.result_tensor_npy_path, result_tensor)
            I.Interpolator.npy_to_dat(self.result_tensor_npy_path, self.result_tensor_dat_path)
            
        return self.stats
            
    def parse_output(self, output):
        """Parse output of GHER DINEOF (extract statistics)

        Args:
            output (str): raw binary output from the Fortran GHER DINEOF

        Returns:
            [str]: Array of extracted statistics
        """
        
        lines = [l.strip() for l in output.decode().split('\n')]
        val_points_num = [int(l.split()[-1]) for l in lines if 'Number of cross validation points' in l][0]
        start_ind = [i + 3 for i, l in enumerate(lines) if 'EOF mode    Expected Error    Iterations made   Convergence achieved' in l][0]  # +3 to skip ____ and empty string
        
        stats = []
        for l in lines[start_ind:]:
            raw_stat = l.split()
            
            if len(raw_stat) < 4:
                break
            else:
                rank = int(raw_stat[0])
                val_rmse = float(raw_stat[1])
                final_iter = int(raw_stat[2])
                conv_error = float(raw_stat[3])
                
                stat = {
                    'rank': rank
                    , 'rmse': val_rmse
                    , 'final_iter': final_iter
                    , 'conv_error': conv_error
                    , 'val_points_num': val_points_num
                }
                stats.append(stat)
                
        return stats

    def predict(self):
        assert hasattr(self, 'result_tensor_npy_path'), ".predict isn't ready, call .fit first."
        reconstructed_tensor = np.load(self.result_tensor_npy_path)
        return reconstructed_tensor

    def get_reconstructed_unified_tensor(self, zero_negative_values=True, apply_log_scale=False,
                                         small_chunk_to_add=.0):
        t = np.load(os.path.abspath(self.result_tensor_npy_path))

        if zero_negative_values:
            t = mutils.zero_negative(t)

        t += small_chunk_to_add

        if apply_log_scale:
            t = mutils.apply_log_scale(t, 0)

        return t

    def get_gapped_unified_tensor(self, zero_negative_values=True, apply_log_scale=False,
                                  small_chunk_to_add=.0):
        t = np.load(os.path.abspath(self.unified_tensor_npy_path))

        if zero_negative_values:
            t = mutils.zero_negative(t)

        t += small_chunk_to_add

        if apply_log_scale:
            t = mutils.apply_log_scale(t, 0)

        return t

    def get_statistics_of_reconstructed_unified_tensor(self, zero_negative_values=True,
                                                       apply_log_scale=False, small_chunk_to_add=.0):
                                                    
        t = self.get_reconstructed_unified_tensor(zero_negative_values, apply_log_scale, small_chunk_to_add)
        return {
            'mean': mutils.get_mean(t),
            'std': mutils.get_std(t),
            'min': mutils.get_min(t),
            'max': mutils.get_max(t)
        }

    def get_statistics_of_gapped_unified_tensor(self, zero_negative_values=False
                                                , apply_log_scale=False, small_chunk_to_add=.0):
                                            
        t = self.get_gapped_unified_tensor(zero_negative_values, apply_log_scale, small_chunk_to_add)
        mask = np.load(self.mask_npy_path)

        return {
            'mean': mutils.get_mean(t),
            'std': mutils.get_std(t),
            'min': mutils.get_min(t),
            'max': mutils.get_max(t),
            'fullness': mutils.calculate_fullness(
                t,
                mutils.form_tensor(mask, t.shape[-1])
            )
        }

    @staticmethod
    def _to_dat(path: str):
        if path[-3:] == 'npy':
            dat_path = path
            dat_path = dat_path[:-3] + 'dat'
            I.Interpolator.npy_to_dat(path, dat_path)
        elif path[-3:] == 'dat':
            dat_path = path
        else:
            raise NotImplementedError()
        return dat_path

    @staticmethod
    def _to_npy(path: str):
        if path[-3:] == 'dat':
            npy_path = path
            npy_path = npy_path[:-3] + 'npy'
            I.Interpolator.dat_to_npy(path, npy_path)
        elif path[-3:] == 'npy':
            npy_path = path
        else:
            raise NotImplementedError()
        return npy_path

    def _construct_dineof_init(self, data, mask, time, output_dir, result_tensor_dat_path):
        """Touch dineof.init and return it's temporary filename"""
        dineof_init = f"""\
            data = ['{data}']
            mask = ['{mask}']
            time = '{time}'
            alpha = {self.alpha}
            numit = {self.numit}
            nev = {self.nev}
            neini = {self.neini}
            ncv = {self.ncv}
            tol = {self.tol}
            nitemax = {self.nitemax}
            toliter = {self.toliter}
            rec = {self.rec}
            eof = {self.eof}
            norm = {self.norm}
            Output = '{output_dir}'
            results = ['{result_tensor_dat_path}']
            seed = {self.seed}
            """

        return bytes(dineof_init, encoding='ascii')
