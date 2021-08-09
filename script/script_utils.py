import os
import re
import pathlib as pb
import warnings
import argparse

import yaml
import numpy as np
import tensorly as tl
from tqdm import tqdm

warnings.filterwarnings("ignore", category=RuntimeWarning) 



def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    config = argparse.Namespace(**config)
    return config


def parse_satellite(base_dir, input_stem=None, output_stem=None, only_years=None):
    """Get input and output dirs from a specified satellite's base_dir"""
    years = os.listdir(base_dir)

    # Filter non year info like raw_data
    year_extractor = re.compile('^\d{4}$')
    years = [y for y in years if year_extractor.match(y)]

    if only_years is not None:
        years = [y for y in years if y in only_years]

    assert input_stem is not None or output_stem is not None
    
    output = None
    if input_stem is not None:
        input_dirs = [os.path.join(base_dir, y, input_stem) for y in years]
        output = input_dirs
    if output_stem is not None:
        output_dirs = [os.path.join(base_dir, y, output_stem) for y in years]
        for o in output_dirs:
            os.makedirs(o, exist_ok=True)
        if output is not None:
            output = output, output_dirs
        else:
            output = output_dirs

    return output


def repeated(func, X, nb_iter=30, random_state=None, verbose=0, mode='bootstrap', **func_kw):
    if random_state is None:
        rng = np.random
    else:
        rng = np.random.RandomState(random_state)
    nb_examples = X.shape[0]
    results = []

    iters = range(nb_iter)
    if verbose > 0:
        iters = tqdm(iters)    
    for i in iters:
        if mode == 'bootstrap':
            Xr = X[rng.randint(0, nb_examples, size=nb_examples)]
        elif mode == 'shuffle':
            ind = np.arange(nb_examples)
            rng.shuffle(ind)
            Xr = X[ind]
        elif mode == 'same':
            Xr = X
        else:
            raise ValueError('unknown mode : {}'.format(mode))
        results.append(func(Xr, **func_kw))
    return results


def bootstrap(*arrays, rng=None, keep_unique_only=True):
    if rng is None:
        rng = np.random
        
    nb_examples = arrays[0].shape[0]    
    bootstrapped_inds = rng.randint(0, nb_examples, size=nb_examples)
    
    if keep_unique_only:
        bootstrapped_inds = np.unique(bootstrapped_inds)
    
    new_arrays = [arr[bootstrapped_inds] for arr in arrays]
    
    if len(new_arrays) > 1:
        output = new_arrays
    else:
        output = new_arrays[0]
        
    return output


def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")
