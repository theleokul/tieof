import sys
import pathlib as pb
import argparse
import copy
import typing as T
import datetime
import multiprocessing as mp

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer

DIR_PATH = pb.Path(sys.argv[0]).resolve().parent
ROOT_PATH = DIR_PATH.parent
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
    parser.add_argument('--decomposition-method', type=str, help='DINEOF, truncSVD, truncHOSVD, HOOI or PARAFAC', default=None)
    parser.add_argument('--nitemax', type=int, default=None)
    parser.add_argument('--refit', type=sutils.str2bool, default=None)
    parser.add_argument('--lat-lon-sep-centering', type=sutils.str2bool, default=None)
    parser.add_argument('--early-stopping', type=sutils.str2bool, default=None)
    parser.add_argument('--random-seed', type=int, default=None)
    parser.add_argument('--trials', type=int, default=None)
    parser.add_argument('--start-trial', type=int, default=None)
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
    if args.decomposition_method.lower() == 'dineof':
        d = DINEOF(
            R=R
            , tensor_shape=args.tensor_shape
            , nitemax=args.nitemax
            , mask=args.mask
            , early_stopping=bool(args.early_stopping)
        )
    else:
        d = DINEOF3(
            R=R
            , tensor_shape=args.tensor_shape
            , decomp_type=args.decomposition_method
            , nitemax=args.nitemax
            , lat_lon_sep_centering=bool(args.lat_lon_sep_centering)
            , mask=args.mask
            , early_stopping=bool(args.early_stopping)
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
    def _main_atom_(X, y, base_stat):
        logger.info('### Calling _main_atom_ ###')

        base_stat = copy.deepcopy(base_stat)
        stats = []
        
        assert args.length > 0
        
        np.random.seed(args.random_seed)
        biner = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='kmeans')
        stratify_y = biner.fit_transform(y[:, None]).flatten().astype(int)
        try:
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=args.val_size, random_state=args.random_seed, stratify=stratify_y)
        except ValueError:
            # The least populated class in y has only 1 member, which is too few. 
            # The minimum number of groups for any class cannot be less than 2. 
            # Thus we dont use stratify option in that case.
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=args.val_size, random_state=args.random_seed)
        logger.info(f'Train points: {y_train.shape[0]}, val points: {y_val.shape[0]}')
        
        base_stat['train_points_num'] = y_train.shape[0]
        base_stat['val_points_num'] = y_val.shape[0]

        val_nrmses = []
        Rs = list(range(args.rank, args.rank + args.length))
        for R in Rs:
            d = get_model(args, R)
            d.fit(X_train, y_train)
            
            rmse = d.rmse(X_val, y_val)
            nrmse = d.nrmse(X_val, y_val)
            
            stat = copy.deepcopy(base_stat)
            stat['rank'] = R
            stat['rmse'] = rmse
            stat['nrmse'] = nrmse
            stat['conv_error'] = d.conv_error
            stat['grad_conv_error'] = d.grad_conv_error
            stat['final_iter'] = d.final_iter
            stats.append(stat)
            
            val_nrmses.append(nrmse)
            
            logger.info(f'Validation rmse: {rmse}')
            logger.info(f'Validation nrmse: {nrmse}')

        best_R = Rs[np.argmin(val_nrmses)]
        best_nrmse = np.min(val_nrmses)
        logger.success(f'Best rank: {best_R}')
        logger.success(f'Lowest validation nrmse: {best_nrmse}')

        if args.refit:  #  or args.length == 0:
            d = get_model(args, best_R)
            d.fit(X, y)
            out = f'{args.out}_{best_R}_{best_nrmse:.4f}.npy'
            np.save(out, d.reconstructed_tensor)
            logger.success(f'Final reconstruction saved to: {out}')
        
        return stats

    # Prepare tensor
    mask = np.load(args.mask).astype(bool)
    tensor = np.load(args.tensor)
    tensor[~mask] = np.nan
    
    # Extract features
    # 2D array, where each row is (lat, lon, day)
    X = np.asarray(np.nonzero(~np.isnan(tensor))).T
    y = tensor[tuple(X.T)]
    
    # Timeline correction
    if args.timeline is not None:
        normalized_timeline = np.load(args.timeline).flatten() - args.first_day
        for i, nt in enumerate(normalized_timeline):
            X[:, 2][X[:, 2] == i] = nt
            
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
            
        base_statb = copy.deepcopy(base_stat)
        base_statb['trial'] = t
        base_statb['known_points_num'] = yb.shape[0]
        base_statb['missing_ratio'] = (mask.sum() * args.tensor_shape[-1] - yb.shape[0]) / (mask.sum() * args.tensor_shape[-1])
            
        statsb = _main_atom_(Xb, yb, base_statb)
        stats.extend(statsb)
        
        df = pd.DataFrame(statsb)
        output_path = f"{args.out}_{args.interpolated_stem}_{'es' if args.early_stopping else 'nes'}_trial_{t:02d}.csv"
        df.to_csv(output_path, index=False)
        
    df = pd.DataFrame(stats)
    output_path = f"{args.out}_{args.interpolated_stem}_{'es' if args.early_stopping else 'nes'}.csv"
    df.to_csv(output_path, index=False)
        

def main():
    config = parse_args()

    is_list = isinstance(config, T.List)
    if is_list and len(config) > 1:
        # Launch each config in parallel
        num_cpus = min(len(config), config[0].process_count)
        logger.info(f'num cpus: {num_cpus} is used for {len(config)} configs.')
        
        with mp.Pool(processes=num_cpus) as pool:
            pool.map(_main_atom, config)
    else:
        if is_list:
            config = config[0] 
        _main_atom(config)

    
if __name__ == '__main__':
    main()
