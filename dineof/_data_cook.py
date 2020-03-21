import os
import re

from tqdm import tqdm
import netCDF4 as nc
import geopandas as gp
from shapely.geometry import MultiPoint
import numpy as np
from sklearn import neighbors
from pathlib import Path
from oct2py import octave

from . import _utils as utils

base_dir = os.path.dirname(os.path.abspath(__file__))
gher_scripts_dir = os.path.join(base_dir, 'gher_scripts')
octave.addpath(gher_scripts_dir)


class DataCook:
    """
        Class that cares about data cooking before fitting with gher dineof
    """
    def __init__(self, shape_file, raw_data_dir, investigated_obj):
        self.shape_file = os.path.abspath(shape_file)
        self.raw_data_dir = os.path.abspath(raw_data_dir)
        self.investigated_obj = investigated_obj

    def get_static_grid_path(self):
        return os.path.join(self.raw_data_dir, 'static_grid')

    def get_static_grid_mask_path(self, extension='npy'):
        return os.path.join(self.get_static_grid_path(), f'mask.{extension}')

    def get_lons_lats_mask(self):
        grid_path = self.get_static_grid_path()
        lons = np.load(os.path.join(grid_path, 'lons.npy'))
        lats = np.load(os.path.join(grid_path, 'lats.npy'))
        mask = np.load(os.path.join(grid_path, 'mask.npy'))

        return lons, lats, mask

    def get_interpolated_path(self):
        return os.path.join(self.raw_data_dir, 'interpolated')

    def get_timeline_path(self, extension='npy'):
        return os.path.join(self.get_interpolated_path(), f'timeline.{extension}')

    def get_timeline(self):
        timeline = np.load(self.get_timeline_path())
        return timeline

    def get_unified_tensor_path(self, extension='npy'):
        return os.path.join(self.get_interpolated_path(), f'unified_tensor.{extension}')

    def touch_static_grid(self, force_static_grid_touch, resolution):
        """Generate spatial grid"""
        static_grid_dir = self.get_static_grid_path()
        lons_path = os.path.join(static_grid_dir, 'lons.npy')
        lats_path = os.path.join(static_grid_dir, 'lats.npy')
        mask_path = os.path.join(static_grid_dir, 'mask.npy')

        if not force_static_grid_touch \
           and os.path.isfile(lons_path) \
           and os.path.isfile(lats_path) \
           and os.path.isfile(mask_path):
            return

        loc = gp.GeoSeries.from_file(self.shape_file)[0]
        lon1, lat1, lon2, lat2 = loc.bounds

        # 111 and 65 are approximate number of kilometers per 1 degree of latitude and longitude accordingly
        spar_lat = abs(int((lat2 - lat1) * 111 / resolution))
        spar_lon = abs(int((lon2 - lon1) * 65 / resolution))
        static_grid_size = spar_lat * spar_lon

        lons, lats = np.meshgrid(np.linspace(lon1, lon2, spar_lon), np.linspace(lat2, lat1, spar_lat))
        grid = list(zip(lons.flatten(), lats.flatten()))
        points = MultiPoint(grid)

        mask = np.zeros(shape=(spar_lon * spar_lat), dtype=np.float)

        print(f'Forming static grid (shape: {spar_lat}x{spar_lon}).')
        for i, p in tqdm(enumerate(points), total=static_grid_size):
            if loc.intersects(p):
                mask[i] = 1
        mask = mask.reshape(spar_lat, spar_lon)

        os.makedirs(static_grid_dir, exist_ok=True)

        np.save(lons_path, lons)
        print(f'lons.npy are created here: {lons_path}')

        np.save(lats_path, lats)
        print(f'lats.npy are created here: {lats_path}')

        np.save(mask_path, mask)
        print(f'mask.npy is created here: {mask_path}')

    def read_raw_data_files(self):
        data_files = utils.ls(self.raw_data_dir)
        utils.guard(all(d.split('.')[-1] == 'nc' for d in data_files), 'NetCDF format is only supported format')

        for raw_data_file in data_files:
            ds = nc.Dataset(raw_data_file, mode='r')

            nav_group = ds.groups['navigation_data']
            # Initially data is masked
            lons = nav_group.variables['longitude'][:]
            # I unmask it for further simpler usage
            lons = np.ma.getdata(lons)
            lats = nav_group.variables['latitude'][:]
            lats = np.ma.getdata(lats)

            geo_group = ds.groups['geophysical_data']
            inv_obj = geo_group.variables[self.investigated_obj][:]

            # Initially mask consists of: False - lake, True - land
            # I want: True - lake, False - land
            inv_obj_mask = np.invert(inv_obj.mask)

            # Unmask and place nan in land's points
            inv_obj.fill_value = np.nan
            inv_obj = inv_obj.filled()

            yield lons, lats, inv_obj, inv_obj_mask, raw_data_file

    def form_cut_mask_on_bounds(self, matrix, bounds):
        b_low, b_high = bounds

        return np.logical_and(matrix > b_low, matrix < b_high)

    def interpolate_raw_data_obj(self, raw_lons, raw_lats, raw_inv_obj, raw_inv_obj_mask):
        # Static grid
        static_grid_dir = self.get_static_grid_path()
        static_lons = np.load(os.path.join(static_grid_dir, 'lons.npy'))
        static_lats = np.load(os.path.join(static_grid_dir, 'lats.npy'))
        static_mask = np.load(os.path.join(static_grid_dir, 'mask.npy')).astype(np.bool)

        # Form mask for raw data from satellite to constrain it on static data
        lons_cut_mask = self.form_cut_mask_on_bounds(raw_lons,
                                                     bounds=(static_lons[:, 0].min(), static_lons[:, -1].max()))
        lats_cut_mask = self.form_cut_mask_on_bounds(raw_lats,
                                                     bounds=(static_lats[-1].min(), static_lats[0].max()))
        cut_mask = np.logical_and(lons_cut_mask, lats_cut_mask)

        # Constrain raw data to newly formed mask
        raw_lons = raw_lons[cut_mask]
        raw_lats = raw_lats[cut_mask]
        raw_inv_obj = raw_inv_obj[cut_mask]
        raw_inv_obj_mask = raw_inv_obj_mask[cut_mask]

        # Get from raw data only known points
        raw_lons_known = raw_lons[raw_inv_obj_mask]
        raw_lats_known = raw_lats[raw_inv_obj_mask]
        raw_lons_lats_known = np.c_[raw_lons_known, raw_lats_known]
        raw_inv_obj_known = raw_inv_obj[raw_inv_obj_mask]

        # Grid on which we will interpolate
        int_lons_lats = np.c_[static_lons.flatten(), static_lats.flatten()]
        int_inv_obj_mask = np.zeros(shape=(int_lons_lats.shape[0]), dtype=np.bool)

        # Defining in which radius to interpolate
        min_grid_distance_lon = abs(static_lons[0][0] - static_lons[0][1])
        min_grid_distance_lat = abs(static_lats[0][0] - static_lats[1][0])
        min_grid_distance = utils.floor_float(np.min([min_grid_distance_lat, min_grid_distance_lon]))

        # Select points to be interpolated that lie in min grid distance radius from raw data
        tree = neighbors.KDTree(raw_lons_lats_known, leaf_size=2)
        for i, int_lon_lat in enumerate(int_lons_lats):
            # If near static node there are raw nodes => we use such static node
            if tree.query_radius(int_lon_lat.reshape(1, -1), r=min_grid_distance, count_only=True)[0] > 0:
                int_inv_obj_mask[i] = True

        int_inv_obj_mask = int_inv_obj_mask.reshape(static_lons.shape)

        # Interpolate filtered nodes, find value based on raw data
        knr = neighbors.KNeighborsRegressor(n_neighbors=3, weights='distance')
        knr.fit(raw_lons_lats_known, raw_inv_obj_known)

        # static_mask - shape of the lake, int_inv_obj_mask - points where we can interpolate
        int_inv_obj_mask = np.logical_and(static_mask, int_inv_obj_mask)
        int_lons_lats_known = int_lons_lats[int_inv_obj_mask.flatten()]
        int_inv_obj_known = knr.predict(int_lons_lats_known)

        # Reconstruct int_inv_obj
        int_inv_obj = np.full(static_lons.shape, np.nan)
        int_inv_obj[int_inv_obj_mask] = int_inv_obj_known

        return int_inv_obj

    def touch_interpolated_data(self, fullness_threshold, remove_low_fullness):
        """
            Interpolate raw data in raw_data_dir to static grid into raw_data_dir/interpolated
        """
        mask = np.load(os.path.join(self.get_static_grid_path(), 'mask.npy')).astype(np.bool)
        interpolated_dir = self.get_interpolated_path()
        os.makedirs(interpolated_dir, exist_ok=True)

        print('Interpolating data.')
        raw_data_files_count = len(utils.ls(self.raw_data_dir))
        for raw_lons, raw_lats, raw_inv_obj, raw_inv_obj_mask, raw_data_file in tqdm(self.read_raw_data_files(),
                                                                                     total=raw_data_files_count):
            raw_data_file_stem = Path(raw_data_file).stem

            if os.path.exists(os.path.join(interpolated_dir, f'{raw_data_file_stem}.npy')):
                continue

            try:
                int_inv_obj = self.interpolate_raw_data_obj(raw_lons, raw_lats, raw_inv_obj, raw_inv_obj_mask)
                fullness = utils.calculate_fullness(int_inv_obj, mask)
            except:
                fullness = 0
                int_inv_obj = np.full(mask.shape, np.nan)

            if fullness_threshold and fullness < fullness_threshold:
                if remove_low_fullness:
                    os.remove(raw_data_file)
                continue

            np.save(os.path.join(interpolated_dir, f'{raw_data_file_stem}.npy'), int_inv_obj)

        print(f'Interpolation is completed, interpolated data is here: {interpolated_dir}')

    def preserve_day_range_only(self, day_range):
        data_files = utils.ls(self.raw_data_dir)
        utils.guard(all(d.split('.')[-1] == 'nc' for d in data_files), 'NetCDF format is only supported format')

        final_data_files = []
        for day in day_range:
            #  Choose all files for specific day
            r_compiler = re.compile(f'^{self.raw_data_dir}/' + r'[a-z]*\d{4}' + f'{day:03d}', re.I)
            filtered_data_files = list(filter(r_compiler.match, data_files))

            final_data_files.extend(filtered_data_files)

        files_to_del = [f for f in data_files if f not in final_data_files]

        for f in files_to_del:
            os.remove(f)

        print(f'Day range: {day_range} is only kept in {self.raw_data_dir}.')

    def preserve_best_day_only(self):
        """
            Preserves the best matrix for one day.

            Filenames in interpolated should be in format *YYYYDDD*.
            .npy extension is only supported.
        """
        static_grid_dir_path = self.get_static_grid_path()
        utils.guard(os.path.isdir(static_grid_dir_path), 'Run touch_static_grid() before this.')

        int_data_dir_path = self.get_interpolated_path()
        utils.guard(os.path.isdir(int_data_dir_path), 'Run touch_interpolated_data() before this.')

        mask_path = self.get_static_grid_mask_path()
        geo_obj_mask = np.load(mask_path)

        data_files = [f for f in utils.ls(int_data_dir_path) if 'unified' not in f and 'timeline' not in f]
        utils.guard(all([f.split('.')[-1] == 'npy' for f in data_files]),
                    'Interpolated chunks in interpolated/ should have .npy ext')

        final_data_files = []
        already_analyzed_days = []
        for f in data_files:
            # Pull day from filename
            day = int(re.search(f'^{int_data_dir_path}/' + r'[a-z]*\d{4}(\d{3})', f, re.I).group(1))
            if day in already_analyzed_days:
                continue
            #  Choose all files for this specific day
            r_compiler = re.compile(f'^{int_data_dir_path}/' + r'[a-z]*\d{4}' + f'{day:03d}', re.I)
            filtered_data_files = list(filter(r_compiler.match, data_files))

            datasets = [np.load(f) for f in filtered_data_files]

            fullness, best_file = utils.calculate_fullness(datasets[0], geo_obj_mask), filtered_data_files[0]
            for i, d in enumerate(datasets[1:], 1):
                new_fullness = utils.calculate_fullness(d, geo_obj_mask)
                if new_fullness > fullness:
                    fullness = new_fullness
                    best_file = filtered_data_files[i]

            final_data_files.append(best_file)
            already_analyzed_days.append(day)

        files_to_del = [f for f in data_files if f not in final_data_files]

        for f in files_to_del:
            os.remove(f)

        print(f'Best day is only kept in {int_data_dir_path}.')

    def touch_unified_tensor(self, move_new_axis_to_end):
        """
            Unify all files from interpolated in 1 tensor and put it
            in the same directory as unified.npy
        """
        int_data_dir_path = self.get_interpolated_path()
        utils.guard(os.path.isdir(int_data_dir_path), 'Run touch_interpolated_data() before this.')

        data_files = [f for f in utils.ls(int_data_dir_path) if 'unified' not in f and 'timeline' not in f]
        utils.guard(all([f.split('.')[-1] == 'npy' for f in data_files]), 'Files in dir_path should have .npy ext')

        unified_tensor = []
        for f in data_files:
            d = np.load(f)
            unified_tensor.append(d)
        unified_tensor = np.array(unified_tensor)

        if move_new_axis_to_end:
            unified_tensor = np.moveaxis(unified_tensor, 0, -1)

        unified_tensor_path = os.path.join(int_data_dir_path, 'unified_tensor.npy')
        np.save(unified_tensor_path, unified_tensor)
        print(f'unified_tensor.npy is created here: {unified_tensor_path}')

    def touch_timeline(self):
        """
            Touch timeline tensor in interpolated

            It is supposed that times are included in filenames
            of interpolated in format *YYYYDDD*
        """
        int_data_dir_path = self.get_interpolated_path()
        utils.guard(os.path.isdir(int_data_dir_path), 'Run touch_interpolated_data() before this.')

        filenames = [f for f in utils.ls(int_data_dir_path) if 'unified' not in f and 'timeline' not in f]
        timeline = []
        for f in filenames:
            m = re.search(r'\d{4}(\d{3})', f)
            if m:
                timeline.append(m.group(1))

        timeline = np.array([timeline], dtype=np.float)

        timeline_path = os.path.join(int_data_dir_path, 'timeline.npy')
        np.save(timeline_path, timeline)
        print(f'timeline.npy is created here: {timeline_path}')

    @staticmethod
    def npy_to_dat(npy_path, dat_path):
        """
            Transform data from .npy to .dat

            npy_path - can be a regular path.
            dat_path - can be a regular path or a directory.
        """
        if os.path.isfile(npy_path):
            utils.guard(npy_path.split('.')[-1] == 'npy', 'npy_path should have .npy ext')
            d = np.load(npy_path)

            if dat_path.split('.')[-1] == 'dat':
                # dat_path is a regular path

                octave.gwrite(dat_path, d)
            else:
                # dat_path is a directory
                os.makedirs(dat_path, exist_ok=True)
                dat_path = os.path.join(dat_path, f'{Path(npy_path).stem}.dat')

                octave.gwrite(dat_path, d)
            print(f'{Path(dat_path).stem}.dat is created here: {dat_path}')
        elif os.path.isdir(npy_path):
            raise Exception('npy_to_dat for directories is not implemented')
        else:
            raise Exception('npy_path should be either directory or regular file')

    @staticmethod
    def dat_to_npy(dat_path, npy_path):
        """
            Transform data from .dat to .npy

            dat_path - can be a regular path.
            npy_path - can be a regular path or a directory.
        """
        if os.path.isfile(dat_path):
            if npy_path.split('.')[-1] == 'npy':
                # npy_path is a regular path

                d = octave.gread(dat_path)
                np.save(npy_path, d)
            else:
                # npy_path is a directory
                os.makedirs(npy_path, exist_ok=True)
                npy_path = os.path.join(npy_path, f'{Path(dat_path).stem}.npy')

                d = octave.gread(dat_path)
                np.save(npy_path, d)
            print(f'{Path(npy_path).stem}.npy is created here: {npy_path}')
        elif os.path.isdir(npy_path):
            raise Exception('dat_to_npy for directories is not implemented')
        else:
            raise Exception('dat_path should be either directory or regular file')
