import os
import re
import pathlib as pb
import warnings
import argparse

import yaml
import numpy as np
import tensorly as tl

warnings.filterwarnings("ignore", category=RuntimeWarning) 


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    config = argparse.Namespace(**config)
    return config


def parse_satellite(base_dir, input_stem, output_stem=None, only_years=None):
    """Get input and output dirs from a specified satellite's base_dir"""
    years = os.listdir(base_dir)

    # Filter non year info like raw_data
    year_extractor = re.compile('^\d{4}$')
    years = [y for y in years if year_extractor.match(y)]

    if only_years is not None:
        years = [y for y in years if y in only_years]

    input_dirs = [os.path.join(base_dir, y, input_stem) for y in years]

    output = input_dirs
    if output_stem is not None:
        output_dirs = [os.path.join(base_dir, y, output_stem) for y in years]
        for o in output_dirs:
            os.makedirs(o, exist_ok=True)
        output = input_dirs, output_dirs

    return output
