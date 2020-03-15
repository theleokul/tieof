import json
import subprocess

from .data_cook import DataCook


def fit(
    data_desc_path='data_desc.json',
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

    with open(data_desc_path, 'r') as f:
        data_desc = json.load(f)

    shape_file = data_desc['shape_file']
    raw_data_dir = data_desc['raw_data_dir']
    investigated_obj = data_desc['investigated_obj']

    dc = DataCook(shape_file, raw_data_dir, investigated_obj)

    dc.touch_static_grid(force_static_grid_touch)
    dc.npy_to_dat(dc.get_static_grid_mask_path(extension='npy'),
                  dc.get_static_grid_path(),
                  force_static_grid_touch)

    if day_range_to_preserve:
        dc.preserve_day_range_only(day_range_to_preserve)

    dc.touch_interpolated_data(fullness_threshold, remove_low_fullness)

    if keep_only_best_day:
        dc.preserve_best_day_only()

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
