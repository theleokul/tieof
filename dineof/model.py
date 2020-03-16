import os
import json
import subprocess

from pathlib import Path

from .data_cook import DataCook


class Dineof:
    def __init__(self, data_desc_path='data_desc.json'):
        with open(data_desc_path, 'r') as f:
            data_desc = json.load(f)

        for k, v in data_desc.items():
            setattr(self, k, v)

        self.dc = DataCook(self.shape_file_path, self.raw_data_dir, self.investigated_obj)

    def fit(
        self,
        fullness_threshold=0.0,
        remove_low_fullness=False,
        force_static_grid_touch=False,
        day_range_to_preserve=range(151, 244),  # (151, 244) - summer
        keep_only_best_day=True
    ):
        """
            Fits the dineof model

            fullness_threshold - minimal proporion of observed data to keep data
            remove_low_fullness - if True: remove raw_inv_obj from raw_data_dir
            force_static_grid_touch - if True: create a static grid if it already exists
            best_day_range_to_preserve - Delete all data for days outside of this range, keep 1 matrix for day
        """
        self.dc.touch_static_grid(force_static_grid_touch)
        DataCook.npy_to_dat(self.dc.get_static_grid_mask_path(extension='npy'),
                            self.dc.get_static_grid_path(),
                            force_static_grid_touch)

        if day_range_to_preserve:
            self.dc.preserve_day_range_only(day_range_to_preserve)

        self.dc.touch_interpolated_data(fullness_threshold, remove_low_fullness)

        if keep_only_best_day:
            self.dc.preserve_best_day_only()

        self.dc.touch_unified_tensor()
        DataCook.npy_to_dat(self.dc.get_unified_tensor_path(extension='npy'),
                            self.dc.get_interpolated_path())

        self.dc.touch_timeline()
        DataCook.npy_to_dat(self.dc.get_timeline_path(extension='npy'),
                            self.dc.get_interpolated_path())

    def predict(self):
        # TODO: dineof_init_path generation based on data_desc_path
        subprocess.call([
            f'{self.dineof_executer}',
            f'{self.dineof_init_path}'
        ])

        # Save output of GHER DINEOF in .npy format
        npy_result_path = os.path.abspath(self.result_path)
        dat_result_path = os.path.join(os.path.abspath(self.output_dir),
                                       f'{Path(npy_result_path).stem}.dat')
        DataCook.dat_to_npy(dat_result_path, npy_result_path)
