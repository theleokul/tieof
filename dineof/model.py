import os
import yaml
import subprocess
import tempfile

import numpy as np
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
from matplotlib.widgets import Slider
from pathlib import Path

from ._data_cook import DataCook
from . import _utils as utils


class Dineof:
    def __init__(self, data_desc_path):
        with open(data_desc_path, 'r') as f:
            data_desc = yaml.safe_load(f)

        for section, section_dict in data_desc.items():
            for k, v in section_dict.items():
                setattr(self, k, v)

        # Initialize object that is responsible for interpolation procedures
        self.dc = DataCook(self.shape_file_path, self.input_dir, self.investigated_obj)

        # Create Output dir if not exists
        os.makedirs(self.output_dir, exist_ok=True)

        # Initialize paths for final result
        self.dat_result_path = os.path.join(
            os.path.abspath(self.output_dir),
            self.dc.get_unified_tensor_path(extension='dat').split('/')[-1]
        )
        self.npy_result_path = os.path.join(
            os.path.abspath(self.output_dir),
            self.dc.get_unified_tensor_path(extension='npy').split('/')[-1]
        )

    def fit(
            self,
            fullness_threshold=0.0,
            remove_low_fullness=False,
            force_static_grid_touch=False,
            day_range_to_preserve=range(151, 244),  # (151, 244) - summer
            keep_only_best_day=True,
            resolution=1,
            move_time_axis_to_end_in_unified_tensor=True
    ):
        """
            Fits the dineof model

            fullness_threshold - minimal proporion of observed data to keep data
            remove_low_fullness - if True: remove raw_inv_obj from raw_data_dir
            force_static_grid_touch - if True: create a static grid if it already exists
            best_day_range_to_preserve - Delete all data for days outside of this range, keep 1 matrix for day
        """
        self.dc.touch_static_grid(force_static_grid_touch, resolution)
        DataCook.npy_to_dat(self.dc.get_static_grid_mask_path(extension='npy'),
                            self.dc.get_static_grid_path())

        if day_range_to_preserve:
            self.dc.preserve_day_range_only(day_range_to_preserve)

        self.dc.touch_interpolated_data(fullness_threshold, remove_low_fullness)

        if keep_only_best_day:
            self.dc.preserve_best_day_only()

        self.dc.touch_unified_tensor(move_time_axis_to_end_in_unified_tensor)
        DataCook.npy_to_dat(self.dc.get_unified_tensor_path(extension='npy'),
                            self.dc.get_interpolated_path())

        self.dc.touch_timeline()
        DataCook.npy_to_dat(self.dc.get_timeline_path(extension='npy'),
                            self.dc.get_interpolated_path())

    def predict(self, zero_negative_in_result_tensor=True):
        with tempfile.NamedTemporaryFile(suffix='.init') as tmp:
            tmp.write(self._construct_dineof_init())
            tmp.seek(0)
            subprocess.call([
                f'{self.dineof_executer}',
                f'{tmp.name}'
            ])

        # Save output of GHER DINEOF in .npy format
        DataCook.dat_to_npy(self.dat_result_path, self.npy_result_path)

        if zero_negative_in_result_tensor:
            result_tensor = np.load(self.npy_result_path)
            result_tensor = utils.zero_negative(result_tensor)
            np.save(self.npy_result_path, result_tensor)


    def _construct_dineof_init(self):
        """Touch dineof.init and return it's temporary filename"""
        dineof_init = f"""\
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
            results = ['{self.dat_result_path}']
            seed = {self.seed}
        """

        return bytes(dineof_init, encoding='ascii')

    def plot(self,
             figsize=(14, 7),
             basemap_width=6 * 1E5,
             basemap_height=6 * 1E5,
             lon_0=106.5,
             lat_0=53.5,
             bar_label='chlor, mg * m^-3',
             clim=(None, None)):

        # Stub
        days = [192]

        fig = plt.figure(figsize=figsize)

        # Load static grid
        lons, lats, mask = self.dc.get_lons_lats_mask()
        mask = mask.astype(np.bool)

        # Get reconstructed unified_tensor
        recovered_unified_tensor = np.load(os.path.abspath(self.npy_result_path))
        recovered_unified_tensor = utils.zero_negative(recovered_unified_tensor)
        recovered_unified_tensor[~np.isnan(recovered_unified_tensor)] = np.log(recovered_unified_tensor[~np.isnan(recovered_unified_tensor)] + 1e-10)
        print(recovered_unified_tensor[~np.isnan(recovered_unified_tensor)].mean())
        # Get gapped unified tensor
        gapped_unified_tensor = self.dc.get_unified_tensor()
        gapped_unified_tensor[~np.isnan(gapped_unified_tensor)] = np.log(gapped_unified_tensor[~np.isnan(gapped_unified_tensor)] + 1e-10)

        # Choose specified days in unified_tensor
        timeline = self.dc.get_timeline()[0]
        print(timeline)
        gapped_recovered_objs = []
        for day in days:
            day_index = np.where(timeline == day)[0][0]
            gapped_recovered_obj = gapped_unified_tensor[:, :, day_index], \
                                   utils.zero_negative(recovered_unified_tensor[:, :, day_index])

            # print(gapped_recovered_obj[1])

            gapped_recovered_objs.append(gapped_recovered_obj)

            # vmin, vmax = gapped_recovered_obj[1][~np.isnan(gapped_recovered_obj[1])].min(), \
            #              gapped_recovered_obj[1][~np.isnan(gapped_recovered_obj[1])].max()

        vmin = recovered_unified_tensor[~np.isnan(recovered_unified_tensor)].min()
        vmax = recovered_unified_tensor[~np.isnan(recovered_unified_tensor)].max()
        print('Recovered vmin, vmax: ', vmin, vmax)

        # Remove extension from shape_file_path (It is required by basemap module)
        shape_file_path_stem = os.path.join(os.path.dirname(self.shape_file_path),
                                            Path(self.shape_file_path).stem)

        # Create plot space
        ax_lst = fig.subplots(1, 2)

        # extent = (lons[mask].min()+1, lons[mask].max()-1, lats[mask].min()+1, lats[mask].max()-1)  # [left, right, bottom, top]

        # Add plot 1
        basemap = Basemap(projection='cyl', resolution='i',
                          llcrnrlon=lons.min(), llcrnrlat=lats.min(), urcrnrlon=lons.max(), urcrnrlat=lats.max(),
                          # width=basemap_width, height=basemap_height,
                          lon_0=lon_0, lat_0=lat_0, ax=ax_lst[0])
        basemap.readshapefile(shape_file_path_stem, name=shape_file_path_stem)

        # Plot inv_obj data
        xi, yi = basemap(lons, lats)
        p_gapped = basemap.imshow(gapped_recovered_objs[0][0], vmin=vmin, vmax=vmax,
                                  origin='upper', interpolation='nearest')

        # end plot 1

        # Add plot 2
        basemap = Basemap(projection='cyl', resolution='i',
                          llcrnrlon=lons.min(), llcrnrlat=lats.min(), urcrnrlon=lons.max(), urcrnrlat=lats.max(),
                          # width=basemap_width, height=basemap_height,
                          lon_0=lon_0, lat_0=lat_0, ax=ax_lst[1])
        basemap.readshapefile(shape_file_path_stem, name=shape_file_path_stem)

        # Plot inv_obj data
        xi, yi = basemap(lons, lats)
        # p_reconstructed = basemap.pcolor(xi, yi, gapped_recovered_objs[0][1], vmin=vmin, vmax=vmax)
        p_reconstructed = basemap.imshow(gapped_recovered_objs[0][1], vmin=vmin, vmax=vmax,
                                         origin='upper')

        # Display colorbar
        fig.subplots_adjust(right=0.8, bottom=0.35)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        sm = cm.ScalarMappable(norm=norm)
        fig.colorbar(sm, cax=cbar_ax)

        # Add slider
        # import ipdb; ipdb.set_trace()

        def day_update(day):
            # Get day index
            day_index = np.where(timeline == day)[0][0]
            gapped_recovered_obj = gapped_unified_tensor[:, :, day_index], \
                                   utils.zero_negative(recovered_unified_tensor[:, :, day_index])

            p_gapped.set_data(gapped_recovered_obj[0])
            p_reconstructed.set_data(gapped_recovered_obj[1])
            plt.draw()

        ax_slider = fig.add_axes([0.1, 0.2, 0.6, 0.05])
        slider = Slider(ax=ax_slider,
                        label='Day',
                        valmin=timeline.min(),
                        valmax=timeline.max(),
                        valinit=timeline.min(),
                        valstep=1,
                        valfmt='%1.0f',
                        color='green')
        slider.on_changed(day_update)

        # end plot 2

        plt.show()
