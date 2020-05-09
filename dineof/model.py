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
        - matplotlib
        """
    )
    PLOT_FEATURE_IS_ENABLED = False

from ._data_cook import DataCook
from . import _utils as utils


class Dineof:
    def __init__(self, data_desc_path):
        with open(data_desc_path, 'r') as f:
            data_desc = yaml.safe_load(f)

        for k, v in data_desc.items():
            setattr(self, k, v)

        # Transform to book for next convenient usage
        self.move_time_axis_in_unified_tensor_to_end = bool(self.move_time_axis_in_unified_tensor_to_end)

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
            resolution=1
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

        self.dc.touch_unified_tensor(self.move_time_axis_in_unified_tensor_to_end)
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

    def get_reconstructed_unified_tensor(self, zero_negative_values=False, apply_log_scale=False,
                                         small_chunk_to_add=.0):
        t = np.load(os.path.abspath(self.npy_result_path))

        if zero_negative_values:
            t = utils.zero_negative(t) + small_chunk_to_add
        else:
            t += small_chunk_to_add

        if apply_log_scale:
            t = utils.apply_log_scale(t, 0)

        return t

    def get_statistics_of_reconstructed_unified_tensor(self, zero_negative_values=False,
                                                       apply_log_scale=False, small_chunk_to_add=.0):
        t = self.get_reconstructed_unified_tensor(zero_negative_values, apply_log_scale, small_chunk_to_add)

        return {
            'mean': utils.get_mean(t),
            'std': utils.get_std(t),
            'min': utils.get_min(t),
            'max': utils.get_max(t)
        }

    def get_statistics_of_gapped_unified_tensor(self, apply_log_scale=False, small_chunk_to_add=.0):
        t = self.dc.get_unified_tensor(apply_log_scale, small_chunk_to_add)
        mask = self.dc.get_mask()

        return {
            'mean': utils.get_mean(t),
            'std': utils.get_std(t),
            'min': utils.get_min(t),
            'max': utils.get_max(t),
            'fullness': utils.calculate_fullness(
                t,
                utils.form_tensor(mask, t.shape[-1 if self.move_time_axis_in_unified_tensor_to_end else 0])
            )
        }

    def plot(self,
             figsize=(14, 7),
             zero_negative_values=True,
             apply_log_scale=True,
             cbar_units='mg * m^-3',
             small_chunk_to_add=1e-10):
        utils.guard(PLOT_FEATURE_IS_ENABLED, 'Your system cannot build plots. Install basemap, matplotlib.')

        fig = plt.figure(figsize=figsize)
        fig.suptitle('Spatio-temporal plot', fontsize=20)
        # Load static grid
        lons, lats = self.dc.get_lons(), self.dc.get_lats()
        lon_mean, lat_mean = lons.mean(), lats.mean()
        # Choose specified days in unified_tensor
        timeline = self.dc.get_timeline()[0]
        # Get tensors to display
        reconstructed_tensor = self.get_reconstructed_unified_tensor(zero_negative_values, apply_log_scale,
                                                                     small_chunk_to_add)
        gapped_tensor = self.dc.get_unified_tensor(apply_log_scale, small_chunk_to_add)
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
        gapped_annotation = gapped_axes.annotate('NOT FOUND', xy=(0.05, 0.9), xycoords='axes fraction',
                                                 fontsize=14, color='red')

        # Draw first reconstructed day
        reconstructed_obj = utils.get_matrix_by_day(timeline[0], timeline, reconstructed_tensor)
        reconstructed_axes = fig.add_axes([0.4, 0.35, 0.47, 0.47])
        reconstructed_axes.set_title('Reconstructed data')
        reconstructed_plot = plot_on(reconstructed_axes, reconstructed_obj)
        reconstructed_annotation = reconstructed_axes.annotate('NOT FOUND', xy=(0.05, 0.9), xycoords='axes fraction',
                                                               fontsize=14, color='red')
        if ~np.all(np.isnan(reconstructed_obj)):
            gapped_annotation.set_visible(False)
            reconstructed_annotation.set_visible(False)

        # Add colorbar
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        sm = mpl.cm.ScalarMappable(norm=norm)
        fig.colorbar(sm, cax=cbar_ax,
                     label=self.investigated_obj + ', ' + f'log({cbar_units})' if apply_log_scale else cbar_units)

        # Add mean and std
        describe_axes = fig.add_axes([0.1, 0.1, 0.7, 0.1], frame_on=False)
        describe_axes.xaxis.set_visible(False)
        describe_axes.yaxis.set_visible(False)
        gapped_statistics_wo_log = self.get_statistics_of_gapped_unified_tensor(False, small_chunk_to_add)
        if apply_log_scale:
            reconstructed_statistics = self.get_statistics_of_reconstructed_unified_tensor(
                zero_negative_values, apply_log_scale, small_chunk_to_add)
            reconstructed_statistics_wo_log = self.get_statistics_of_reconstructed_unified_tensor(
                zero_negative_values, small_chunk_to_add=small_chunk_to_add)
            describe_data = [
                [
                    reconstructed_statistics_wo_log['mean'],
                    reconstructed_statistics_wo_log['std'],
                    reconstructed_statistics_wo_log['min'],
                    reconstructed_statistics_wo_log['max'],
                    gapped_statistics_wo_log['fullness']
                ],
                [
                    reconstructed_statistics['mean'],
                    reconstructed_statistics['std'],
                    reconstructed_statistics['min'],
                    reconstructed_statistics['max'],
                    np.nan
                ]
            ]
            describe_axes.table(cellText=describe_data,
                                rowLabels=['1', 'log'], colLabels=['mean', 'std', 'min', 'max', 'fullness'],
                                cellLoc='right', rowLoc='right', colLoc='right', edges='vertical')
        else:
            describe_data = [
                [
                    utils.get_mean(reconstructed_tensor),
                    utils.get_std(reconstructed_tensor),
                    utils.get_min(reconstructed_tensor),
                    utils.get_max(reconstructed_tensor),
                    gapped_statistics_wo_log['fullness']
                ]
            ]
            describe_axes.table(cellText=describe_data,
                                rowLabels=['1'], colLabels=['mean', 'std', 'min', 'max', 'fullness'],
                                cellLoc='right', rowLoc='right', colLoc='right', edges='vertical')

        def day_update(day):
            gapped_obj = utils.get_matrix_by_day(day, timeline, gapped_tensor)
            reconstructed_obj = utils.get_matrix_by_day(day, timeline, reconstructed_tensor)
            if np.all(np.isnan(reconstructed_obj)):
                gapped_annotation.set_visible(True)
                reconstructed_annotation.set_visible(True)
            else:
                gapped_annotation.set_visible(False)
                reconstructed_annotation.set_visible(False)

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
