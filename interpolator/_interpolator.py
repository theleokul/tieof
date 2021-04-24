import os
import re
import sys
from pathlib import Path

from loguru import logger
from tqdm import tqdm
import netCDF4 as nc
import geopandas as gp
from shapely.geometry import MultiPoint
import numpy as np
from sklearn import neighbors
from oct2py import octave

DIR_PATH = Path(__file__).resolve().parent
GHER_SCRIPTS_DIR_PATH = DIR_PATH / 'gher_scripts'
sys.path.append(str(DIR_PATH))
import interpolator_utils as iutils


octave.addpath(str(GHER_SCRIPTS_DIR_PATH))
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'  # HACK: May be useful when NetCDF4 dataset reading raises an exception



class Interpolator:

    def __init__(
        self
        , shape_file: str  # path tp .shp file
        , raw_data_dir: str  # input_dir filepath with .nc datasets you desire to interpolate
        , investigated_obj: str  # something to extract out of .nc files ('chlor_a' for example)
        , investigated_obj__threshold=np.inf

        # All the files related to interpolation are created inside raw_data_dir
        , static_grid_stem: str='static_grid' 
        , interpolated_stem: str='interpolated'
        , unified_tensor_stem: str='unified_tensor'
        , timeline_stem: str='timeline'
    ):

        self.shape_file = os.path.abspath(shape_file)
        self.raw_data_dir = os.path.abspath(raw_data_dir)
        self.investigated_obj = investigated_obj
        self.investigated_obj__threshold = investigated_obj__threshold
        self.static_grid_stem = static_grid_stem
        self.interpolated_stem = interpolated_stem
        self.unified_tensor_stem = unified_tensor_stem
        self.timeline_stem = timeline_stem

    def fit(
            self,
            interpolation_strategy='radius',
            fullness_threshold=0.0,
            remove_low_fullness=False,
            force_static_grid_touch=False,
            day_bounds_to_preserve=(151, 244),  # (151, 244) - summer
            keep_only_best_day=True,
            resolution=1,
            move_time_axis_in_unified_tensor_to_end=True,
            create_dat_copies=True
    ):
        """
            Fit the interpolator, i.e. create static_grid, interpolated tensors and timeline.

            Arguments:
            
                * fullness_threshold - minimal proporion of observed data to keep data
                * remove_low_fullness - if True: remove raw_inv_obj from raw_data_dir
                * force_static_grid_touch - if True: create a static grid if it already exists
                * day_bounds_to_preserve - Delete all data for days outside of this bounds.
                                          WARNING: Actually deletes the data with rm system calls.
                * create_dat_copies - by default only .npy tensors are created, this option duplicates the content 
                                      to .dat files (typically used with fortran software)
        """

        self.touch_static_grid(force_static_grid_touch, resolution)
        self.npy_to_dat(self.get_static_grid_mask_path(extension='npy'),
                        self.get_static_grid_path())

        if day_bounds_to_preserve is not None:
            assert len(day_bounds_to_preserve) == 2
            day_range_to_preserve = range(day_bounds_to_preserve[0], day_bounds_to_preserve[1])
            self.preserve_day_range_only(day_range_to_preserve)

        self.touch_interpolated_data(fullness_threshold, remove_low_fullness, interpolation_strategy)

        if keep_only_best_day:
            self.preserve_best_day_only()

        self.touch_unified_tensor(move_time_axis_in_unified_tensor_to_end)
        self.touch_timeline()

        if create_dat_copies:
            self.npy_to_dat(self.get_unified_tensor_path(extension='npy'),
                            self.get_interpolated_path())
            self.npy_to_dat(self.get_timeline_path(extension='npy'),
                            self.get_interpolated_path())

    def get_static_grid_path(self):
        return os.path.join(self.raw_data_dir, self.static_grid_stem)

    def get_static_grid_mask_path(self, extension='npy'):
        return os.path.join(self.get_static_grid_path(), f'mask.{extension}')

    def get_lons(self):
        grid_path = self.get_static_grid_path()
        lons = np.load(os.path.join(grid_path, 'lons.npy'))
        return lons

    def get_lats(self):
        grid_path = self.get_static_grid_path()
        lats = np.load(os.path.join(grid_path, 'lats.npy'))
        return lats

    def get_mask(self):
        grid_path = self.get_static_grid_path()
        mask = np.load(os.path.join(grid_path, 'mask.npy'))
        return mask

    def get_lons_lats_mask(self):
        return self.get_lons(), self.get_lats(), self.get_mask()

    def get_interpolated_path(self):
        return os.path.join(self.raw_data_dir, self.interpolated_stem)

    def get_timeline_path(self, extension='npy'):
        return os.path.join(self.get_interpolated_path(), f'{self.timeline_stem}.{extension}')

    def get_timeline(self):
        timeline = np.load(self.get_timeline_path())
        return timeline

    def get_unified_tensor_path(self, extension='npy'):
        return os.path.join(self.get_interpolated_path(), f'{self.unified_tensor_stem}.{extension}')

    def get_unified_tensor(self, apply_log_scale=False, small_chunk_to_add=0):
        unified_tensor = np.load(self.get_unified_tensor_path()) + small_chunk_to_add
        if apply_log_scale:
            unified_tensor = iutils.apply_log_scale(unified_tensor)

        return unified_tensor

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

        logger.info(f'Forming static grid (shape: {spar_lat}x{spar_lon}).')
        for i, p in tqdm(enumerate(points), total=static_grid_size):
            if loc.intersects(p):
                mask[i] = 1
        mask = mask.reshape(spar_lat, spar_lon)

        os.makedirs(static_grid_dir, exist_ok=True)

        np.save(lons_path, lons)
        logger.info(f'lons.npy are created here: {lons_path}')

        np.save(lats_path, lats)
        logger.info(f'lats.npy are created here: {lats_path}')

        np.save(mask_path, mask)
        logger.info(f'mask.npy is created here: {mask_path}')

    def read_raw_data_files(self):
        data_files = iutils.ls(self.raw_data_dir)
        iutils.guard(all(d.split('.')[-1] == 'nc' for d in data_files), 'NetCDF format is only supported format')

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

    def interpolate_raw_data_obj(self, raw_lons, raw_lats, raw_inv_obj, raw_inv_obj_mask, interpolation_strategy):
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

        logger.info(f'Original investigated object statistics: \
            \nmin: {raw_inv_obj_known.min()}, \nmax: {raw_inv_obj_known.max()}, \
            \nmean: {raw_inv_obj_known.mean()}, \nmedian: {np.median(raw_inv_obj_known)}')

        if np.isfinite(self.investigated_obj__threshold):
            raw_inv_obj_known = np.clip(raw_inv_obj_known, 0, self.investigated_obj__threshold)
            logger.info(f'Original investigated object is clipped to: 0. - {self.investigated_obj__threshold}')

        # Grid on which we will interpolate
        int_lons_lats = np.c_[static_lons.flatten(), static_lats.flatten()]
        int_inv_obj_mask = np.zeros(shape=(int_lons_lats.shape[0]), dtype=np.bool)

        # Defining in which radius to interpolate
        # It is actually euclidean metric, because 1 component equal 0
        # I do not consider diagonal points, because according to a + b < c, they are higher
        min_grid_distance_lon = abs(static_lons[0][0] - static_lons[0][1])
        min_grid_distance_lat = abs(static_lats[0][0] - static_lats[1][0])
        min_grid_distance = iutils.floor_float(np.min([min_grid_distance_lat, min_grid_distance_lon]))

        # Select points to be interpolated that lie in min grid distance radius from raw data
        tree = neighbors.KDTree(raw_lons_lats_known, leaf_size=2)
        for i, int_lon_lat in enumerate(int_lons_lats):
            # If near static node there are raw nodes => we use such static node
            if tree.query_radius(int_lon_lat.reshape(1, -1), r=min_grid_distance, count_only=True)[0] > 0:
                int_inv_obj_mask[i] = True

        int_inv_obj_mask = int_inv_obj_mask.reshape(static_lons.shape)

        # Interpolate filtered nodes, find value based on raw data
        # knr = neighbors.KNeighborsRegressor(n_neighbors=3, weights='distance')
        if interpolation_strategy == 'radius':
            knr = neighbors.RadiusNeighborsRegressor(radius=min_grid_distance * 5., weights='distance')   # ~ 5 km
        elif interpolation_strategy == 'neighbours':
            knr = neighbors.KNeighborsRegressor(n_neighbors=3, weights='distance')
        else:
            raise NotImplementedError()

        knr.fit(raw_lons_lats_known, raw_inv_obj_known)

        # static_mask - shape of the lake, int_inv_obj_mask - points where we can interpolate
        int_inv_obj_mask = np.logical_and(static_mask, int_inv_obj_mask)
        int_lons_lats_known = int_lons_lats[int_inv_obj_mask.flatten()]
        int_inv_obj_known = knr.predict(int_lons_lats_known)

        # HACK: Tp be safe that indeed borders are correct
        int_inv_obj_known = np.clip(int_inv_obj_known, 0, self.investigated_obj__threshold)
        assert int_inv_obj_known.min() >= 0 \
            and int_inv_obj_known.max() <= self.investigated_obj__threshold

        # Reconstruct int_inv_obj
        int_inv_obj = np.full(static_lons.shape, np.nan)
        int_inv_obj[int_inv_obj_mask] = int_inv_obj_known

        return int_inv_obj

    def touch_interpolated_data(
        self
        , fullness_threshold
        , remove_low_fullness
        , interpolation_strategy  # Either radius or neighbours
    ):

        """
            Interpolate raw data in raw_data_dir to static grid into raw_data_dir/interpolated
        """
        mask = np.load(os.path.join(self.get_static_grid_path(), 'mask.npy')).astype(np.bool)
        interpolated_dir = self.get_interpolated_path()
        os.makedirs(interpolated_dir, exist_ok=True)

        logger.info('Interpolating data.')
        raw_data_files_count = len(iutils.ls(self.raw_data_dir))
        for i, (raw_lons, raw_lats, raw_inv_obj, raw_inv_obj_mask, raw_data_file) in tqdm(enumerate(self.read_raw_data_files()),
                                                                                          total=raw_data_files_count):
            raw_data_file_stem = Path(raw_data_file).stem

            if os.path.exists(os.path.join(interpolated_dir, f'{raw_data_file_stem}.npy')):
                continue

            try:
                int_inv_obj = self.interpolate_raw_data_obj(raw_lons, raw_lats, raw_inv_obj, raw_inv_obj_mask, interpolation_strategy)
                fullness = iutils.calculate_fullness(int_inv_obj, mask)
            except Exception as e:
                logger.warning(f'{i} is empty. {e}')
                fullness = 0
                int_inv_obj = np.full(mask.shape, np.nan)

            if fullness_threshold and fullness < fullness_threshold:
                if remove_low_fullness:
                    os.remove(raw_data_file)
                continue

            np.save(os.path.join(interpolated_dir, f'{raw_data_file_stem}.npy'), int_inv_obj)

        logger.success(f'Interpolation is completed, interpolated data is here: {interpolated_dir}')

    def preserve_day_range_only(self, day_range):
        data_files = iutils.ls(self.raw_data_dir)
        iutils.guard(all(d.split('.')[-1] == 'nc' for d in data_files), 'NetCDF format is only supported format')

        final_data_files = []
        for day in day_range:
            #  Choose all files for specific day
            r_compiler = re.compile(f'^{self.raw_data_dir}/' + r'[a-z]*\d{4}' + f'{day:03d}', re.I)
            filtered_data_files = list(filter(r_compiler.match, data_files))

            final_data_files.extend(filtered_data_files)

        files_to_del = [f for f in data_files if f not in final_data_files]

        for f in files_to_del:
            os.remove(f)

        logger.info(f'Day range: {day_range} is only kept in {self.raw_data_dir}.')

    def preserve_best_day_only(self):
        """
            Preserves the best matrix for one day.

            Filenames in interpolated should be in format *YYYYDDD*.
            .npy extension is only supported.
        """
        static_grid_dir_path = self.get_static_grid_path()
        iutils.guard(os.path.isdir(static_grid_dir_path), 'Run touch_static_grid() before this.')

        int_data_dir_path = self.get_interpolated_path()
        iutils.guard(os.path.isdir(int_data_dir_path), 'Run touch_interpolated_data() before this.')

        mask_path = self.get_static_grid_mask_path()
        geo_obj_mask = np.load(mask_path)

        data_files = [f for f in iutils.ls(int_data_dir_path) if self.unified_tensor_stem not in f and self.timeline_stem not in f]
        iutils.guard(all([f.split('.')[-1] == 'npy' for f in data_files]),
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

            fullness, best_file = iutils.calculate_fullness(datasets[0], geo_obj_mask), filtered_data_files[0]
            for i, d in enumerate(datasets[1:], 1):
                new_fullness = iutils.calculate_fullness(d, geo_obj_mask)
                if new_fullness > fullness:
                    fullness = new_fullness
                    best_file = filtered_data_files[i]

            final_data_files.append(best_file)
            already_analyzed_days.append(day)

        files_to_del = [f for f in data_files if f not in final_data_files]

        for f in files_to_del:
            os.remove(f)

        logger.info(f'Best day is only kept in {int_data_dir_path}.')

    def touch_unified_tensor(self, move_new_axis_to_end):
        """
            Unify all files from interpolated in 1 tensor and put it
            in the same directory as unified.npy
        """
        int_data_dir_path = self.get_interpolated_path()
        iutils.guard(os.path.isdir(int_data_dir_path), 'Run touch_interpolated_data() before this.')

        data_files = [f for f in iutils.ls(int_data_dir_path) if self.unified_tensor_stem not in f and self.timeline_stem not in f]
        iutils.guard(all([f.split('.')[-1] == 'npy' for f in data_files]), 'Files in dir_path should have .npy ext')

        unified_tensor = []
        for f in data_files:
            d = np.load(f)
            unified_tensor.append(d)
        unified_tensor = np.array(unified_tensor)

        if move_new_axis_to_end:
            unified_tensor = np.moveaxis(unified_tensor, 0, -1)

        unified_tensor_path = self.get_unified_tensor_path(extension='npy')
        np.save(unified_tensor_path, unified_tensor)
        logger.info(f'unified_tensor is created here: {unified_tensor_path}')

    def touch_timeline(self):
        """
            Touch timeline tensor in interpolated

            It is supposed that times are included in filenames
            of interpolated in format *YYYYDDD*
        """
        int_data_dir_path = self.get_interpolated_path()
        iutils.guard(os.path.isdir(int_data_dir_path), 'Run touch_interpolated_data() before this.')

        filenames = [f for f in iutils.ls(int_data_dir_path) if self.unified_tensor_stem not in f and self.timeline_stem not in f]
        timeline = []
        for f in filenames:
            m = re.search(r'\d{4}(\d{3})', f)
            if m:
                timeline.append(m.group(1))

        timeline = np.array([timeline], dtype=np.float)

        timeline_path = self.get_timeline_path(extension='npy')
        np.save(timeline_path, timeline)
        logger.info(f'timeline is created here: {timeline_path}')

    @staticmethod
    def npy_to_dat(npy_path, dat_path):
        """
            Transform data from .npy to .dat

            npy_path - can be a regular path.
            dat_path - can be a regular path or a directory.
        """
        if os.path.isfile(npy_path):
            iutils.guard(npy_path.split('.')[-1] == 'npy', 'npy_path should have .npy ext')
            d = np.load(npy_path)

            if dat_path.split('.')[-1] == 'dat':
                # dat_path is a regular path

                octave.gwrite(dat_path, d)
            else:
                # dat_path is a directory
                os.makedirs(dat_path, exist_ok=True)
                dat_path = os.path.join(dat_path, f'{Path(npy_path).stem}.dat')
                octave.gwrite(dat_path, d)
            logger.info(f'{Path(dat_path).stem}.dat is created here: {dat_path}')
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
            logger.info(f'{Path(npy_path).stem}.npy is created here: {npy_path}')
        elif os.path.isdir(npy_path):
            raise Exception('dat_to_npy for directories is not implemented')
        else:
            raise Exception('dat_path should be either directory or regular file')
