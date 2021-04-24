import os
import sys
import pathlib as pb
import argparse
import copy
import typing as T
import datetime

import ray
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer

DIR_PATH = pb.Path(sys.argv[0]).resolve().parent
ROOT_PATH = DIR_PATH.parent
# HACK: For ray to be able to import from parent directory
os.environ["PYTHONPATH"] = str(ROOT_PATH) + ":" + os.environ.get("PYTHONPATH", "")
sys.path.append(str(ROOT_PATH))
import script.script_utils as sutils
from model import DINEOF, DINEOF3



def parse_args():
    parser = argparse.ArgumentParser(description='DINEOF3 main entry.')

    # One tensor to reconstruct scenario (All default values are Nones to not override config values in vain)
    parser.add_argument('-c', '--config', help='Base config that could be overriden with other cli args', type=str, required=True)

    parser.add_argument('-t', '--tensor', type=str, help='Path to numpy representation of a tensor to reconstruct', default=None)
    parser.add_argument('-O', '--out', type=str, help='Save path for the reconstruction', default=None)
    parser.add_argument('-T', '--timeline', type=str, help='Path to numpy representation of a timeline', default=None)
    parser.add_argument('-m', '--mask', type=str, help='Path to numpy representation of a mask', default=None)

    parser.add_argument('--first-day', type=int, default=None)
    parser.add_argument('-R', '--rank', type=int, help='Rank to use in the decomposition algorithm', default=None)
    parser.add_argument('-L', '--length', type=int, help='Validation length', default=None)
    parser.add_argument('--tensor-shape', nargs=3, type=int, help='Tensor shape', default=None)
    parser.add_argument('--decomposition-method', type=str, help='truncSVD, truncHOSVD, HOOI or PARAFAC', default=None)
    parser.add_argument('--nitemax', type=int, default=None)
    parser.add_argument('--refit', type=bool, default=None)
    parser.add_argument('--lat-lon-sep-centering', type=bool, default=None)
    parser.add_argument('--random-seed', type=int, default=None)
    parser.add_argument('--val-size', type=float, default=None)
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
            # NOTE: Out should be appended with rank, validation error and .npy extension later
            sub_config.out = str(pb.Path(o) / f'{base_config.unified_tensor_stem}_{base_config.decomposition_method.lower()}')
            sub_config.timeline = str(pb.Path(i) / base_config.interpolated_stem / f'{base_config.timeline_stem}.npy')
            sub_config.mask = str(pb.Path(i) / base_config.static_grid_stem / 'mask.npy')
            
            config.append(sub_config)

    return config


def get_model(args, R):
    if args.decomposition_method.lower() == 'truncsvd':
        d = DINEOF(
            R=R
            , tensor_shape=args.tensor_shape
            , nitemax=args.nitemax
            , mask=args.mask
        )
    else:
        d = DINEOF3(
            R=R
            , tensor_shape=args.tensor_shape
            , decomp_type=args.decomposition_method
            , nitemax=args.nitemax
            , lat_lon_sep_centering=args.lat_lon_sep_centering
            , mask=args.mask
        )
    
    return d


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
    def _main_atom_():
        mask = np.load(args.mask).astype(bool)
        tensor = np.load(args.tensor)
        tensor[~mask] = np.nan

        # 2D array, where each row is (lat, lon, day)
        X = np.asarray(np.nonzero(~np.isnan(tensor))).T
        y = tensor[tuple(X.T)]

        # Timeline correction
        if args.timeline is not None:
            logger.info('Timeline correction gets its hands dirty...')
            normalized_timeline = np.load(args.timeline).flatten() - args.first_day
            logger.info(f'Normalized timeline: {list(normalized_timeline)}')
            for i, nt in enumerate(normalized_timeline):
                X[:, 2][X[:, 2] == i] = nt

        logger.info(f'Missing ratio: {(mask.sum() * args.tensor_shape[-1] - y.shape[0]) / (mask.sum() * args.tensor_shape[-1])}')

        if args.length > 0:
            np.random.seed(args.random_seed)
            biner = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='kmeans')
            stratify_y = biner.fit_transform(y[:, None]).flatten().astype(int)
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=args.val_size, random_state=args.random_seed, stratify=stratify_y)
            logger.info(f'Train points: {y_train.shape[0]}, val points: {y_val.shape[0]}')

            val_errors = []
            Rs = list(range(args.rank, args.rank + args.length))
            for R in Rs:
                d = get_model(args, R)
                d.fit(X_train, y_train)
                val_errors.append(-d.score(X_val, y_val) * y_val.std())
                logger.info(f'Validation error: {val_errors[-1]}')

            best_R = Rs[np.argmin(val_errors)]
            best_val_error = np.min(val_errors)
            logger.success(f'Best rank: {best_R}')
            logger.success(f'Lowest validation error: {best_val_error}')
        else:
            best_R = args.rank
            best_val_error = None

        if args.refit or args.length == 0:
            d = get_model(args, best_R)
            d.fit(X, y)

        if best_val_error is not None:
            out = f'{args.out}_{best_R}_{best_val_error:.4f}.npy'
        else:
            out = f'{args.out}_{best_R}.npy'

        np.save(out, d.reconstructed_tensor)
        logger.success(f'Final reconstruction saved to: {out}')

    _main_atom_()


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
