import os
import yaml
import subprocess
import tempfile

from .data_cook import DataCook


class Dineof:
    def __init__(self, data_desc_path):
        with open(data_desc_path, 'r') as f:
            data_desc = yaml.safe_load(f)

        for section, section_dict in data_desc.items():
            for k, v in section_dict.items():
                setattr(self, k, v)

        self.dc = DataCook(self.shape_file_path, self.input_dir, self.investigated_obj)

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
        with tempfile.NamedTemporaryFile() as tmp:
            tmp.write(self.construct_dineof_init())
            subprocess.call([
                f'{self.dineof_executer}',
                f'{tmp.name}'
            ])

        # Save output of GHER DINEOF in .npy format
        npy_result_path = os.path.abspath(self.result_path)
        dat_result_path = os.path.join(
            os.path.abspath(self.output_dir),
            f'{self.dc.get_unified_tensor_path(extension="dat").split("/")[-1]}'
        )
        DataCook.dat_to_npy(dat_result_path, npy_result_path)

    def construct_dineof_init(self):
        """Touch dineof.init and return it's temporary filename"""
        dineof_init = f"""
            data = ['{self.dc.get_unified_tensor_path(extension='dat')}']
            mask = ['{self.dc.get_static_grid_mask_path(extension='dat')}']
            time = '{self.dc.get_timeline_path(extension='dat')}'
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
            Output = '{os.path.abspath(self.output_dir)}'
            results = ['{os.path.join(os.path.abspath(self.output_dir),
                         self.dc.get_unified_tensor_path(extension='dat').split('/')[-1])}']
            seed = {self.seed}
        """

        return bytes(dineof_init, encoding='utf-8')
