"""
    CLI for GHER DINEOF.
"""

import sys
import pathlib as pb
import argparse

import numpy as np


DIR_PATH = pb.Path(__file__).resolve().parent
ROOT_PATH = DIR_PATH.parent
sys.path.append(str(DIR_PATH))
import script_utils as sutils
sys.path.append(str(ROOT_PATH))
from model import DINEOFGHER



def parse_args():
    parser = argparse.ArgumentParser(description='DINEOF3 main entry.')
    parser.add_argument('-c', '--config', type=str, required=True)
    parser.add_argument('-t', '--tensor', type=str, required=True)
    parser.add_argument('-m', '--mask', type=str, required=True)
    parser.add_argument('--timeline', type=str, required=True)
    parser.add_argument('-O', '--output-dir', type=str, required=True)
    parser.add_argument('--output-stem', type=str, default='unified_tensor')
    parser.add_argument('-z', '--zero-negative', type=bool, default=True)
    parser.add_argument('-S', '--satellite', type=str, default=None)
    parser.add_argument('-S', '--satellite-descriptor'
                        , type=str
                        , help='Path to .csv file with key-value pairs that maps satellites to base dirs'
                        , default='/home/kulikov/dineof3/test/satellite_descriptor.csv')
    args = parser.parse_args()

    if args.satellite is not None:
        

    return args


def main():
    args = parse_args()
    config = sutils.load_config(args.config)
    d = DINEOFGHER(config)

    d.fit(
        args.tensor  # Correct order of axes: (lat, lon, t)
        , args.mask
        , args.timeline
        
        , args.output_dir
        , args.output_stem

        , args.zero_negative
    )

if __name__ == '__main__':
    main()
