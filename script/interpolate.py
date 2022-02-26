import os
import re
import sys
import copy
import typing as T
import pathlib as pb
import argparse as ap
import datetime

import ray
import yaml
import numpy as np
import pandas as pd
from loguru import logger

DIR_PATH = pb.Path(__file__).resolve().parent
ROOT_PATH = DIR_PATH.parent
# HACK: For ray to be able to import from parent directory
os.environ["PYTHONPATH"] = str(ROOT_PATH) + ":" + os.environ.get("PYTHONPATH", "")
sys.path.append(str(ROOT_PATH))
import script.script_utils as sutils
from interpolator import _interpolator as I



def parse_args() -> T.Union[ap.Namespace, T.List[ap.Namespace]]:
    parser = ap.ArgumentParser(description='Inglorious Interpolator.')
    parser.add_argument('-c', '--config', help='Base config that could be overriden with other cli args', type=str, required=True)
    parser.add_argument('-I', '--input-dir', type=str, default=None)
    parser.add_argument('-S', '--satellite', type=str, default=None)
    parser.add_argument('--satellite-descriptor'
                        , type=str
                        , help='Path to .csv file with key-value pairs that maps satellites to base dirs'
                        , default='../supp/satellite_descriptor.csv')
    parser.add_argument('--only-years', type=str, nargs='+', default=None)
    parser.add_argument('--input-stem', type=str, default='Input')
    parser.add_argument('--static-grid-stem', type=str, default='static_grid')
    parser.add_argument('--interpolated-stem', type=str, default='interpolated')
    parser.add_argument('--unified-tensor-stem', type=str, default='unified_tensor')
    parser.add_argument('--timeline-stem', type=str, default='timeline')
    parser.add_argument('--interpolation-strategy', type=str, default='radius', choices=['radius', 'neighbours'])
    parser.add_argument('-p', '--process-count', type=int, default=np.inf)
    parser.add_argument('--logs', type=str, default='./logs')
    args = parser.parse_args()

    config = sutils.load_config(args.config)
    setattr(config, 'process_count', args.process_count)
    setattr(config, 'logs', args.logs)
    os.makedirs(config.logs, exist_ok=True)
    
    if args.input_dir is not None:
        setattr(config, 'input_dir', args.input_dir)
    if args.static_grid_stem is not None:
        setattr(config, 'static_grid_stem', args.static_grid_stem)
    if args.interpolated_stem is not None:
        setattr(config, 'interpolated_stem', args.interpolated_stem)
    if args.unified_tensor_stem is not None:
        setattr(config, 'unified_tensor_stem', args.unified_tensor_stem)
    if args.timeline_stem is not None:
        setattr(config, 'timeline_stem', args.timeline_stem)
    if args.interpolation_strategy is not None:
        setattr(config, 'interpolation_strategy', args.interpolation_strategy)
    if args.only_years is not None:
        setattr(config, 'only_years', args.only_years)
        
    if args.satellite is not None:
        assert args.satellite_descriptor is not None
        df = pd.read_csv(args.satellite_descriptor)
        satellite_base_dir = df[df.satellite == args.satellite].base_dir.iloc[0]
        input_dirs = \
            sutils.parse_satellite(satellite_base_dir
                                   , input_stem=args.input_stem
                                   , only_years=config.only_years)

        base_config = copy.deepcopy(config)
        config = []
        for i in input_dirs:
            sub_config = copy.deepcopy(base_config)
            sub_config.input_dir = i
            config.append(sub_config)

    return config


def _main_atom(config):
    from loguru import logger  # HACK: Allows to separate loggers between ray processes
    year = config.input_dir.split('/')[-2]
    dt = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    logger.add(
        str(pb.Path(config.logs) / (dt + '-' + year + '.log'))
        , format='{time} {level} {message}'
    )

    logger.info(f'Config: {config}')

    interp = I.Interpolator(
        shape_file=config.shape_file_path
        , raw_data_dir=config.input_dir
        , investigated_obj=config.investigated_obj
        , static_grid_stem=config.static_grid_stem
        , interpolated_stem=config.interpolated_stem
        , unified_tensor_stem=config.unified_tensor_stem
        , timeline_stem=config.timeline_stem
        , investigated_obj__threshold=config.investigated_obj__threshold if hasattr(config, 'investigated_obj__threshold') else np.inf
    )
    interp.fit(
        interpolation_strategy=config.interpolation_strategy,
        fullness_threshold=config.fullness_threshold,
        remove_low_fullness=config.remove_low_fullness,
        force_static_grid_touch=config.force_static_grid_touch,
        day_bounds_to_preserve=config.day_bounds_to_preserve,
        keep_only_best_day=config.keep_only_best_day,
        resolution=config.resolution,
        move_time_axis_in_unified_tensor_to_end=config.move_time_axis_in_unified_tensor_to_end,
        create_dat_copies=config.create_dat_copies
    )


@ray.remote
def _main_atom_ray(*args, **kwargs):
    return _main_atom(*args, **kwargs)


def main():
    config = parse_args()

    is_list = isinstance(config, T.List)
    if is_list and len(config) > 1:
        # Launch each config in parallel
        num_cpus = min(len(config), config[0].process_count)
        logger.info(f'num cpus: {num_cpus} is used for {len(config)} configs.')
        ray.init(num_cpus=num_cpus)
        ray.get([_main_atom_ray.remote(c) for c in config])
        ray.shutdown()
    else:
        if is_list:
            config = config[0] 
        _main_atom(config)


if __name__ == '__main__':
    main()
