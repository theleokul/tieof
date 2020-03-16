import os
import yaml
import subprocess
import tempfile

from pathlib import Path
import numpy as np
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt

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
            tmp.write(self.dc.construct_dineof_init())
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

    def plot(self,
             day,
             basemap_width=6 * 1E5,
             basemap_height=6 * 1E5,
             lon_0=106.5,
             lat_0=53.5,
             bar_label='chlor, mg * m^-3',
             clim=(None, None)):
        unified_tensor = np.load(os.path.abspath(self.result_path))
        timeline = self.dc.get_timeline()[0]
        day_index = np.where(timeline == day)[0][0]
        inv_obj = unified_tensor[:, :, day_index]

        # Load static grid
        lons, lats, _ = self.dc.get_lons_lats_mask()

        # Prepare background of the plot
        basemap = Basemap(projection='lcc', resolution='i',
                          width=basemap_width, height=basemap_height,
                          lon_0=lon_0, lat_0=lat_0)

        shape_filename = os.path.join(os.path.dirname(self.shape_file_path),
                                      Path(self.shape_file_path).stem)
        basemap.readshapefile(shape_filename, name=shape_filename)

        # Plot inv_obj data
        xi, yi = basemap(lons, lats)
        basemap.pcolor(xi, yi, inv_obj, vmin=clim[0], vmax=clim[1])

        # Display colorbar
        cbar = basemap.colorbar()
        cbar.ax.get_yaxis().labelpad = 10
        cbar.ax.set_ylabel(bar_label, rotation=90)

        plt.show()
