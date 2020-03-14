import os
import re

from tqdm import tqdm
import netCDF4 as nc
import geopandas as gp
from shapely.geometry import MultiPoint
import numpy as np
import numpy.ma as ma
from sklearn import neighbors
from pathlib import Path
from oct2py import octave

from . import utils

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

    def get_interpolated_path(self):
        return os.path.join(self.raw_data_dir, 'interpolated')

    def get_timeline_path(self, extension='npy'):
        return os.path.join(self.get_interpolated_path(), f'timeline.{extension}')

    def get_unified_tensor_path(self, extension='npy'):
        return os.path.join(self.get_interpolated_path(), f'unified.{extension}')

    def touch_static_grid(self, resolution=1):
        """Generate spatial grid"""
        loc = gp.GeoSeries.from_file(self.shape_file)[0]
        lon1, lat1, lon2, lat2 = loc.bounds

        # 111 and 65 are approximate number of kilometers per 1 degree of latitude and longitude accordingly
        spar_lat = abs(int((lat2 - lat1) * 111 / resolution))
        spar_lon = abs(int((lon2 - lon1) * 65 / resolution))

        lons, lats = np.meshgrid(np.linspace(lon1, lon2, spar_lon), np.linspace(lat2, lat1, spar_lat))
        grid = list(zip(lons.flatten(), lats.flatten()))
        points = MultiPoint(grid)

        mask = np.zeros(shape=(spar_lon * spar_lat), dtype=np.float)

        print(f'Forming static grid (shape: {spar_lat}x{spar_lon}).')
        for i, p in tqdm(enumerate(points)):
            if loc.intersects(p):
                mask[i] = 1
        mask = mask.reshape(spar_lat, spar_lon)

        static_grid_dir = self.get_static_grid_path()
        os.makedirs(static_grid_dir, exist_ok=True)

        lons_path = os.path.join(static_grid_dir, 'lons.npy')
        np.save(lons_path, lons)
        print(f'lons.npy are created here: {lons_path}')

        lats_path = os.path.join(static_grid_dir, 'lats.npy')
        np.save(lats_path, lats)
        print(f'lats.npy are created here: {lats_path}')

        mask_path = os.path.join(static_grid_dir, 'mask.npy')
        np.save(mask_path, mask)
        print(f'mask.npy is created here: {mask_path}')

    def read_raw_data_files(self):
        data_files = utils.ls(self.raw_data_dir)
        utils.guard(all(d.split('.')[-1] == 'nc' for d in data_files), 'NetCDF format is only supported format')

        for raw_data_file in data_files:
            ds = nc.Dataset(raw_data_file, mode='r')

            nav_group = ds.groups['navigation_data']
            lons = nav_group.variables['longitude'][:]
            lats = nav_group.variables['latitude'][:]

            geo_group = ds.groups['geophysical_data']
            inv_obj = geo_group.variables[self.investigated_obj][:]

            ma_lons = ma.array(lons, mask=inv_obj.mask, fill_value=inv_obj.fill_value)
            ma_lats = ma.array(lats, mask=inv_obj.mask, fill_value=inv_obj.fill_value)

            yield ma_lons, ma_lats, inv_obj, raw_data_file

    def interpolate_raw_data_obj(self, raw_lons, raw_lats, raw_inv_obj, static_lons, static_lats):
        # Raw data from satellite
        raw_X = np.c_[raw_lons.compressed(), raw_lats.compressed()]
        raw_y = raw_inv_obj.compressed()

        # Grid on which we will interpolate (np.c_ implicitly removes mask and acts like np.getdata)
        int_X = np.c_[static_lons.flatten(), static_lats.flatten()]
        int_X_mask = np.ones(shape=(int_X.shape[0]), dtype=np.bool)

        # Defining in which radius to interpolate
        min_grid_distance_lon = abs(ma.getdata(static_lons)[0][0] - ma.getdata(static_lons)[0][1])
        min_grid_distance_lat = abs(ma.getdata(static_lats)[0][0] - ma.getdata(static_lats)[1][0])
        min_grid_distance = utils.floor_float(np.min([min_grid_distance_lat, min_grid_distance_lon]))

        # Select points to be interpolated that lie in min grid distance radius from raw data
        tree = neighbors.KDTree(raw_X, leaf_size=2)
        for i, x in enumerate(int_X):
            # If near static node there are raw nodes => we use such static node
            if tree.query_radius(x.reshape(1, -1), r=min_grid_distance, count_only=True)[0] > 0:
                int_X_mask[i] = False
        int_X_mask = int_X_mask.reshape(static_lons.shape)

        # Interpolate filtered nodes, find value based on raw data
        knr = neighbors.KNeighborsRegressor(n_neighbors=3, weights='distance')
        T = np.c_[ma.getdata(static_lons).flatten(), ma.getdata(static_lats).flatten()]
        int_inv_obj_mask = np.logical_or(static_lons.mask, int_X_mask)
        int_inv_obj = knr.fit(raw_X, raw_y).predict(T)
        int_inv_obj = int_inv_obj.reshape(static_lons.shape)
        # int_inv_obj_mask: 1 - land, 0 - water
        int_inv_obj[int_inv_obj_mask] = np.nan

        return int_inv_obj

    def touch_interpolated_data(self, fullness_threshold, remove_low_fullness):
        """
            Interpolate raw data in raw_data_dir to static grid into raw_data_dir/interpolated
        """
        static_grid_dir = self.get_static_grid_path()
        utils.guard(os.path.isdir(static_grid_dir), 'static_grid_dir is not created')

        static_lons = np.load(os.path.join(static_grid_dir, 'lons.npy'))
        static_lats = np.load(os.path.join(static_grid_dir, 'lats.npy'))
        mask = np.load(os.path.join(static_grid_dir, 'mask.npy'))

        static_lons = ma.array(static_lons, mask=~(mask.astype(np.bool)), dtype=np.float)
        static_lats = ma.array(static_lats, mask=~(mask.astype(np.bool)), dtype=np.float)

        interpolated_dir = self.get_interpolated_path()
        os.makedirs(interpolated_dir, exist_ok=True)

        print('Interpolating data.')
        for raw_lons, raw_lats, raw_inv_obj, raw_data_file in tqdm(self.read_raw_data_files()):
            raw_data_file_stem = raw_data_file.split('/')[-1]

            if os.path.exists(os.path.join(interpolated_dir, f'{raw_data_file_stem}.npy')):
                continue

            try:
                int_inv_obj = self.interpolate_raw_data_obj(raw_lons, raw_lats, raw_inv_obj, static_lons, static_lats)
                fullness = utils.calculate_fullness(int_inv_obj, mask)
            except:
                fullness = 0
                # Fill int_inv_obj with unknown values, i.e. nan
                int_inv_obj = np.full(raw_lons.shape, np.nan)

            if fullness_threshold and fullness < fullness_threshold:
                if remove_low_fullness:
                    os.remove(raw_data_file)
                continue

            np.save(os.path.join(interpolated_dir, f'{raw_data_file_stem}.npy'), int_inv_obj)

        print(f'Interpolation is completed, interpolated data is here: {interpolated_dir}')

    def preserve_best_day_only(self, day_range):
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
        utils.guard(all([f.split('.')[-1] == 'npy' for f in data_files]), 'Files in dir_path should have .npy ext')

        final_data_files = []
        for day in day_range:
            #  Choose all files for specific day
            r_compiler = re.compile(f'^{int_data_dir_path}/' + r'[a-b]*\d{4}' + f'{day:03d}', re.I)
            filtered_data_files = list(filter(r_compiler.match, data_files))

            # If no data for that day is provided, skip
            if not filtered_data_files:
                continue

            datasets = [np.load(f) for f in filtered_data_files]

            fullness, best_file = utils.calculate_fullness(datasets[0], geo_obj_mask), filtered_data_files[0]
            for i, d in enumerate(datasets[1:]):
                new_fullness = utils.calculate_fullness(d, geo_obj_mask)
                if new_fullness > fullness:
                    fullness = new_fullness
                    best_file = filtered_data_files[i]

            final_data_files.append(best_file)

        files_to_del = [f for f in data_files if f not in final_data_files]

        for f in files_to_del:
            os.remove(f)

    def touch_unified_tensor(self, move_new_axis_to_end=True):
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

    def npy_to_dat(self, npy_path, dat_path):
        """
            Transform data from .npy to .dat

            npy_path - can be a regular path.
            dat_path - can be a regular path or a directory.
        """
        if os.path.isfile(npy_path):
            utils.guard(npy_path.split('.')[-1] == 'npy', 'npy_path should have .npy ext')
            d = np.load(npy_path)

            if dat_path.split('.')[-1] == 'dat':
                pass
                # dat_path is a regular path
                octave.gwrite(dat_path, d)
            else:
                # dat_path is a directory
                os.makedirs(dat_path, exist_ok=True)
                dat_path = os.path.join(dat_path, f'{Path(npy_path).stem}.dat')
                print(dat_path)
                octave.gwrite(dat_path, d)
        elif os.path.isdir(npy_path):
            raise Exception('npy_to_dat for directories is not implemented')
        else:
            raise Exception('npy_path should be either directory or regular file')
