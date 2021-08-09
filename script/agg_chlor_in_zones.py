import os
import re
import sys
import copy
import argparse as ap
import pathlib as pb
from pathlib import Path
import typing as T
import datetime
from functools import reduce

import ray
from tqdm import tqdm
import numpy as np
import pandas as pd
from loguru import logger

DIR_PATH = pb.Path(__file__).resolve().parent
ROOT_PATH = DIR_PATH.parent
# HACK: For ray to be able to import from parent directory
os.environ["PYTHONPATH"] = str(ROOT_PATH) + ":" + os.environ.get("PYTHONPATH", "")
sys.path.append(str(ROOT_PATH))
import script.script_utils as sutils



def parse_args() -> T.Union[ap.Namespace, T.List[ap.Namespace]]:
    parser = ap.ArgumentParser(description='Aggregator for zones.')
    parser.add_argument('-t', '--tensor-path', type=str, default=None, help='Reconstructed tensor path')
    parser.add_argument('-z', '--zones-dir', type=str, default=None, help='Dir to masks')
    parser.add_argument('--unify-masks', type=sutils.str2bool, default=0, help='Whether to unify masks within zoens-dir or not.')
    parser.add_argument('-o', '--output-path', type=str, help='Output path', required=True)
    parser.add_argument('-S', '--satellite', type=str, default=None)
    parser.add_argument('--satellite-descriptor'
                        , type=str
                        , help='Path to .csv file with key-value pairs that maps satellites to base dirs'
                        , default='../supp/satellite_descriptor.csv')
    parser.add_argument('--only-years', type=str, nargs='+', default=None)
    parser.add_argument('--output-stem', type=str, default='Output_5kmradius_thresh2')
    parser.add_argument('--unified-tensor-re', type=str, default=r'^unified_tensor_hooi_10_\d.\d{4}.npy$')
    parser.add_argument('-p', '--process-count', type=int, default=np.inf)
    parser.add_argument('--logs', type=str, default='./logs')
    args = parser.parse_args()

    os.makedirs(args.logs, exist_ok=True)
    
    if args.tensor_path is None:
        assert args.satellite is not None
        assert args.satellite_descriptor is not None
        df = pd.read_csv(args.satellite_descriptor)
        satellite_base_dir = df[df.satellite == args.satellite].base_dir.iloc[0]
        output_dirs = \
            sutils.parse_satellite(satellite_base_dir
                                   , output_stem=args.output_stem
                                   , only_years=args.only_years)

        base_config = copy.deepcopy(args)
        config = []
        for o in output_dirs:
            sub_config = copy.deepcopy(base_config)
            # unified_tensor_hooi_10_0.3605.npy
            unified_tensor_extractor = re.compile(args.unified_tensor_re)
            unified_tensor_names = os.listdir(o)
            try:
                unified_tensor_name = [u for u in unified_tensor_names if unified_tensor_extractor.match(u)][0]
            except IndexError as e:
                logger.debug(f'Tensor not found in: {o}')
                raise e
            sub_config.tensor_path = str(pb.Path(o) / unified_tensor_name)
            config.append(sub_config)

    return config


def _main_atom(args):
    from loguru import logger  # HACK: Allows to separate loggers between ray processes
    year = args.tensor_path.split('/')[-3]
    dt = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    logger.add(
        str(pb.Path(args.logs) / (dt + '-' + year + '.log'))
        , format='{time} {level} {message}'
    )

    logger.info(f'Config: {args}')
    
    tensor_path = args.tensor_path
    output_path = args.output_path
    
    os.makedirs(output_path, exist_ok=True)
    zones_dir = args.zones_dir

    tensor = np.load(tensor_path)
    tensor[tensor < 0] = 0
    t_dim_size = tensor.shape[-1]
    
    if zones_dir is not None:
        mask_names = os.listdir(zones_dir)
        mask_paths = [os.path.join(zones_dir, m) for m in mask_names if m.split('.')[-1] == 'npy']
        masks = [np.load(p).astype(np.bool) for p in mask_paths]
        if args.unify_masks:
            mask_names = [reduce(lambda x, y: pb.Path(x).stem + '__' + pb.Path(y).stem, mask_names)]
            masks = [reduce(lambda x, y: x | y, masks)]
    else:
        mask_names = ['global']
        masks = [None]
        
    df = pd.DataFrame()
    for mask, mask_name in tqdm(zip(masks, mask_names), total=len(masks)):
        chlor_means = []
        
        for day_ind in range(t_dim_size):
            filtered_chlor = tensor[:, :, day_ind]
            if mask is not None:
                filtered_chlor = filtered_chlor[mask]
            filtered_chlor = filtered_chlor[~np.isnan(filtered_chlor)]
            chlor_means.append(filtered_chlor.mean())
        
        chlor_means = np.array(chlor_means)
        df[mask_name] = chlor_means

    df.to_csv(str(pb.Path(output_path) / f'{year}.csv'), index=False)
    
    
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
