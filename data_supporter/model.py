import os
import json

from data_preparer import DataPreparer


def fit(data_desc_path):
    with open(data_desc_path, 'r') as f:
        data_desc = json.load(f)

    shape_file = os.path.abspath(data_desc['shape_file'])
    raw_data_dir = os.path.abspath(data_desc['raw_data_dir'])
    investigated_obj = os.path.abspath(data_desc['investigated_obj'])

    dp = DataPreparer(shape_file, raw_data_dir, investigated_obj)

    dp.build_static_grid()
    dp.npy_to_mat(dp.get_static_grid_mask_path(extension='npy'), dp.get_static_grid_path())
    dp.mat_to_dat(dp.get_static_grid_mask_path(extension='mat'), dp.get_static_grid_path())

    dp.touch_interpolated_data(fullness_threshold=0.05)

    dp.preserve_best_day_only()

    dp.touch_unified_tensor()
    dp.npy_to_mat(dp.get_unified_path(extension='npy'), dp.get_interpolated_path())
    dp.mat_to_dat(dp.get_unified_path(extension='mat'), dp.get_interpolated_path())

    dp.touch_timeline()
    dp.npy_to_mat(dp.get_timeline_path(extension='npy'), dp.get_interpolated_path())
    dp.mat_to_dat(dp.get_timeline_path(extension='mat'), dp.get_interpolated_path())
