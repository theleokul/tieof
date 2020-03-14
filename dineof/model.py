import os
import json
import subprocess

from .data_cook import DataCook


def fit(
    data_desc_path,
    fullness_threshold=0.0,
    remove_low_fullness=False,
    day_range_to_preserve=(151, 244)  # (151, 244) - summer
):
    """
        Fits the dineof model

        fullness_threshold - minimal proporion of observed data to keep data
        remove_low_fullness - if True: remove raw_inv_obj from raw_data_dir
        day_range_to_preserve - Deletes all data for days outside of this range
    """

    with open(data_desc_path, 'r') as f:
        data_desc = json.load(f)

    shape_file = os.path.abspath(data_desc['shape_file'])
    raw_data_dir = os.path.abspath(data_desc['raw_data_dir'])
    investigated_obj = os.path.abspath(data_desc['investigated_obj'])

    dc = DataCook(shape_file, raw_data_dir, investigated_obj)

    dc.build_static_grid()
    dc.npy_to_dat(dc.get_static_grid_mask_path(extension='npy'), dc.get_static_grid_path())

    dc.touch_interpolated_data(fullness_threshold, remove_low_fullness)

    dc.preserve_best_day_only(day_range_to_preserve)

    dc.touch_unified_tensor()
    dc.npy_to_dat(dc.get_unified_tensor_path(extension='npy'), dc.get_interpolated_path())

    dc.touch_timeline()
    dc.npy_to_dat(dc.get_timeline_path(extension='npy'), dc.get_interpolated_path())


def predict(dineof_executer, dineof_init_path):
    subprocess.call([
        f'{dineof_executer}',
        f'{dineof_init_path}'
    ])


def fit_predict(
    data_desc_path,
    dineof_executer,
    dineof_init_path,
    fullness_threshold=0.0,
    remove_low_fullness=False,
    day_range_to_preserve=(151, 244)  # (151, 244) - summer
):
    fit(data_desc_path, fullness_threshold, remove_low_fullness, day_range_to_preserve)
    predict(dineof_executer, dineof_init_path)
