import os
import argparse
from pathlib import Path
from collections import defaultdict

from tqdm import tqdm
import numpy as np
import pandas as pd
from loguru import logger


parser = argparse.ArgumentParser(description='chlor miner for tataink')
parser.add_argument('-t', '--tensor-path', type=str, help='Reconstructed tensor path', required=True)
parser.add_argument('-o', '--output-path', type=str, help='Output path', required=True)
parser.add_argument('-z', '--zones-dir', type=str, help='Dir to masks')


args = parser.parse_args()
tensor_path = args.tensor_path
output_path = args.output_path
os.makedirs(str(Path(output_path).resolve().parent), exist_ok=True)
zones_dir = args.zones_dir


tensor = np.load(tensor_path)
tensor[tensor < 0] = 0
t_dim_size = tensor.shape[-1]


means_df = pd.DataFrame()
for root, dirs, files in os.walk(zones_dir, topdown=False):
    rff = [os.path.join(root, f) for f in files if f.split('.')[-1] == 'npy']
    masks = [np.load(rf) for rf in rff]

    unified_mask = masks[0].astype(np.bool)
    for mask in tqdm(masks[1:]):
        unified_mask |= mask.astype(np.bool)
        
    chlor_means = []
    for day_ind in range(t_dim_size):
        mat = tensor[:, :, day_ind]
        filtered_chlor = mat[unified_mask]
        filtered_chlor = filtered_chlor[~np.isnan(filtered_chlor)]

        chlor_means.append() += filtered_chlor.mean()
    
    chlor_means = np.array(chlor_means)
    means_df['agg_chlor'] = chlor_means

means_df.to_csv(output_path, index=False)
