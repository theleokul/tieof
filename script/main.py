"""
    CLI for GHER DINEOF.
"""

import os
import sys
import pathlib as pb
import argparse
import copy
import datetime
import typing as T

import ray
import numpy as np
import pandas as pd
from loguru import logger

DIR_PATH = pb.Path(sys.argv[0]).resolve().parent
ROOT_PATH = DIR_PATH.parent
# HACK: For ray to be able to import from parent directory
os.environ["PYTHONPATH"] = str(ROOT_PATH) + ":" + os.environ.get("PYTHONPATH", "")
sys.path.append(str(ROOT_PATH))
import script.script_utils as sutils
from model import DINEOFGHER, model_utils as mutils



def parse_args():
    parser = argparse.ArgumentParser(description='DINEOF3 main entry.')

    # One tensor to reconstruct scenario (All default values are Nones to not override config values in vain)
    parser.add_argument('-c', '--config', help='Base config that could be overriden with other cli args', type=str, required=True)

    parser.add_argument('-t', '--tensor', type=str, help='Path to numpy representation of a tensor to reconstruct', default=None)
    parser.add_argument('-O', '--out', type=str, help='Save path for the reconstruction', default=None)
    parser.add_argument('-T', '--timeline', type=str, help='Path to numpy representation of a timeline', default=None)
    parser.add_argument('-m', '--mask', type=str, help='Path to numpy representation of a mask', default=None)
    parser.add_argument('--tensor-shape', nargs=3, type=int, help='Tensor shape', default=None)
    parser.add_argument('--trials', type=int, default=None)
    parser.add_argument('--start-trial', type=int, default=None)
    parser.add_argument('--logs', type=str, default=None)

    # Aggretated scenario args (Basically multiplies config with all tensors of the satellite with different tensors, out etc...)
    parser.add_argument('-S', '--satellite', type=str, default=None)
    parser.add_argument('--satellite-descriptor'
                        , type=str
                        , help='Path to .csv file with key-value pairs that maps satellites to base dirs'
                        , default=None)
    parser.add_argument('--only-years', type=str, nargs='+', help='Used only with --satellite to reconstruct only specified years.', default=None)
    parser.add_argument('--input-stem', type=str, default=None)
    parser.add_argument('--output-stem', type=str, default=None)
    parser.add_argument('--static-grid-stem', type=str, default=None)
    parser.add_argument('--interpolated-stem', type=str, default=None)
    parser.add_argument('--unified-tensor-stem', type=str, default=None)
    parser.add_argument('--timeline-stem', type=str, default=None)
    parser.add_argument('-p', '--process-count', type=int, default=np.inf)

    args = parser.parse_args()

    config = sutils.load_config(args.config)

    # Update parameters from config with CLI
    args.config = None  # HACK: To avoid setting config property inside config Namespace
    for k, v in vars(args).items():
        if v is not None:
            setattr(config, k, v)

    # Expand config to configs if aggregating (satellite) option is provided
    if config.satellite is not None:
        if args.satellite_descriptor is not None:
            setattr(config, 'satellite_descriptor', args.satellite_descriptor)

        df = pd.read_csv(config.satellite_descriptor)
        satellite_base_dir = df[df.satellite == args.satellite].base_dir.iloc[0]

        input_dirs, output_dirs = \
            sutils.parse_satellite(satellite_base_dir
                                   , input_stem=config.input_stem
                                   , output_stem=config.output_stem
                                   , only_years=config.only_years)

        base_config = copy.deepcopy(config)
        config = []
        for i, o in zip(input_dirs, output_dirs):
            sub_config = copy.deepcopy(base_config)

            sub_config.input_dir = i
            sub_config.output_dir = o
            sub_config.tensor = str(pb.Path(i) / base_config.interpolated_stem / f'{base_config.unified_tensor_stem}.npy')
            sub_config.out = str(pb.Path(o) / f'{base_config.unified_tensor_stem}_dineofgher')
            sub_config.output_stem = base_config.unified_tensor_stem
            sub_config.timeline = str(pb.Path(i) / base_config.interpolated_stem / f'{base_config.timeline_stem}.npy')
            sub_config.mask = str(pb.Path(i) / base_config.static_grid_stem / 'mask.npy')
            
            config.append(sub_config)

    return config


def _main_atom(args):
    from loguru import logger  # HACK: Allows to separate loggers between ray processes

    year = args.input_dir.split('/')[-2]
    dt = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    logger.add(
        str(pb.Path(args.logs) / (dt + '-' + year + '.log'))
        , format='{time} {level} {message}'
    )

    logger.info(f'Config: {args}')

    # Wrap contents into another function to catch exceptions to log files
    @logger.catch(reraise=True)
    def _main_atom_(X, y, base_stat):
        logger.info('### Calling _main_atom_ ###')

        base_stat = copy.deepcopy(base_stat)
        
        d = DINEOFGHER(args)
        stats = d.fit(
            unified_tensor_path=args.tensor  # Correct order of axes: (lat, lon, t)
            , mask_path=args.mask
            , timeline_path=args.timeline
            
            , output_dir=args.output_dir
            , output_stem=args.output_stem

            , zero_negative_in_result_tensor=True
        )
        
        stats = [{
            **s
            , **base_stat
            , 'train_points_num': base_stat['known_points_num'] - s['val_points_num']
            , 'nrmse': np.nan
            , 'grad_conv_error': np.nan
        } for s in stats]

        logger.success(str(stats))
        
        return stats

    # Prepare tensor
    mask = np.load(args.mask).astype(bool)
    tensor = np.load(args.tensor)
    tensor[~mask] = np.nan
    
    # Extract features
    # 2D array, where each row is (lat, lon, day)
    X = np.asarray(np.nonzero(~np.isnan(tensor))).T
    y = tensor[tuple(X.T)]
            
    # Build base_stat
    base_stat = {
        'year': year
        , 'masked_points_num': mask.sum() * args.tensor_shape[-1]
    }

    stats = []
    rng = np.random.RandomState(args.random_seed)
    for t in range(args.start_trial, args.trials):
        if t < 1:
            Xb, yb = copy.deepcopy(X), copy.deepcopy(y)
        else:
            Xb, yb = sutils.bootstrap(X, y, rng=rng, keep_unique_only=True)
            
        args.tensor = str(pb.Path(args.input_dir) / args.interpolated_stem / 'tmp_unified_tensorb.npy')
        tensorb = mutils.tensorify(Xb, yb, tensor.shape)
        np.save(args.tensor, tensorb)
            
        base_statb = copy.deepcopy(base_stat)
        base_statb['trial'] = t
        base_statb['known_points_num'] = yb.shape[0]
        base_statb['missing_ratio'] = (mask.sum() * args.tensor_shape[-1] - yb.shape[0]) / (mask.sum() * args.tensor_shape[-1])
        
        statsb = _main_atom_(Xb, yb, base_statb)
        stats.extend(statsb)
        
        df = pd.DataFrame(statsb)
        output_path = f"{args.out}_{args.interpolated_stem}_nes_trial_{t:02d}.csv"
        df.to_csv(output_path, index=False)
        
    df = pd.DataFrame(stats)
    output_path = f"{args.out}_{args.interpolated_stem}_nes.csv"
    df.to_csv(output_path, index=False)
        

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
        # _main_atom(config[0])
    else:
        if is_list:
            config = config[0] 
        _main_atom(config)


# def main():
#     args = parse_args()
#     config = sutils.load_config(args.config)
#     d = DINEOFGHER(config)

#     d.fit(
#         args.tensor  # Correct order of axes: (lat, lon, t)
#         , args.mask
#         , args.timeline
        
#         , args.output_dir
#         , args.output_stem

#         , args.zero_negative
#     )

if __name__ == '__main__':
    main()
