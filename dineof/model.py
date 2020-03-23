import os
import yaml
import subprocess
import tempfile

import numpy as np

try:
    from mpl_toolkits.basemap import Basemap
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from matplotlib.widgets import Slider
    PLOT_FEATURE_IS_ENABLED = True
except ModuleNotFoundError:
    print(
        """
        NOTE: If you want to use plot features of this module, install these modules:
        - basemap
        - matplotlibs
        """
    )
    PLOT_FEATURE_IS_ENABLED = False

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

    def get_reconstructed_unified_tensor(self, zero_negative_values=False, apply_log_scale=False):
        t = np.load(os.path.abspath(self.npy_result_path))

        if zero_negative_values:
            t = utils.zero_negative(t)

        if apply_log_scale:
            t = utils.apply_log_scale(t)

        return t

    def plot(self,
             figsize=(14, 7),
             zero_negative_values=True,
             apply_log_scale=True,
             cbar_label='chlor, mg * m^-3'):
        utils.guard(PLOT_FEATURE_IS_ENABLED, 'Your system cannot build plots')

        fig = plt.figure(figsize=figsize)
        fig.suptitle('Spatio-temporal plot', fontsize=20)
        # Load static grid
        lons, lats, _ = self.dc.get_lons_lats_mask()
        lon_mean, lat_mean = lons.mean(), lats.mean()
        # Choose specified days in unified_tensor
        timeline = self.dc.get_timeline()[0]
        # Get tensors to display
        reconstructed_tensor = self.get_reconstructed_unified_tensor(zero_negative_values, apply_log_scale)
        gapped_tensor = self.dc.get_unified_tensor(apply_log_scale)
        # Calculate color edges
        vmin = utils.get_min(reconstructed_tensor)
        vmax = utils.get_max(reconstructed_tensor)
        # Remove extension from shape_file_path (It is required by basemap module)
        clipped_shape_file_path = utils.remove_extension(self.shape_file_path)

        def plot_on(axes, data):
            basemap = Basemap(projection='cyl', resolution='i',
                              llcrnrlon=lons.min(), llcrnrlat=lats.min(), urcrnrlon=lons.max(), urcrnrlat=lats.max(),
                              lon_0=lon_mean, lat_0=lat_mean, ax=axes)
            basemap.readshapefile(clipped_shape_file_path, name=clipped_shape_file_path)
            plot = basemap.imshow(data, vmin=vmin, vmax=vmax, origin='upper')
            return plot

        # Draw first gapped day
        gapped_obj = utils.get_matrix_by_day(timeline[0], timeline, gapped_tensor)
        gapped_axes = fig.add_axes([0, 0.35, 0.47, 0.47])
        gapped_axes.set_title('Gapped data')
        gapped_plot = plot_on(gapped_axes, gapped_obj)

        # Draw first reconstructed day
        reconstructed_obj = utils.get_matrix_by_day(timeline[0], timeline, reconstructed_tensor)
        reconstructed_axes = fig.add_axes([0.4, 0.35, 0.47, 0.47])
        reconstructed_axes.set_title('Reconstructed data')
        reconstructed_plot = plot_on(reconstructed_axes, reconstructed_obj)

        # Add colorbar
        # fig.subplots_adjust(right=0.8, bottom=0.35)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7], label=cbar_label)
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        sm = mpl.cm.ScalarMappable(norm=norm)
        fig.colorbar(sm, cax=cbar_ax)

        def day_update(day):
            gapped_obj = utils.get_matrix_by_day(day, timeline, gapped_tensor)
            reconstructed_obj = utils.get_matrix_by_day(day, timeline, reconstructed_tensor)
            gapped_plot.set_data(gapped_obj)
            reconstructed_plot.set_data(reconstructed_obj)
            plt.draw()

        # Add slider
        slider_axes = fig.add_axes([0.1, 0.15, 0.7, 0.05])
        slider = Slider(ax=slider_axes,
                        label='Day',
                        valmin=timeline.min(),
                        valmax=timeline.max(),
                        valinit=timeline.min(),
                        valstep=1,
                        valfmt='%1.0f',
                        color='green')
        slider.on_changed(day_update)
        plt.show()
