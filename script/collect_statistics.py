import os
import re
import pathlib as pb

import pandas as pd
import tqdm

SATELLITE_DATA_DIRPATH = '/mss3/baikal/kulikov'
SATELLITES = ['modis_aqua', 'modis_terra', 'seawifs']
OUTPUT_SUFFIXES = [
    '3neighbours'
    , '3neighbours_thresh2'
    , '5kmradius'
    , '5kmradius_thresh2'
]
RESULT_PATH = '/mss3/baikal/kulikov/statistics.csv'
MAX_TRIAL = 29
CONV_MODE = ['es', 'nes']
METHODS = [
    'dineofgher'
    , 'dineof'
    , 'parafac'
    , 'hooi'
    , 'trunchosvd'
]



def main():
    
    df_unified = None
    for sat in tqdm.tqdm(SATELLITES):
        sat_dirpath = pb.Path(SATELLITE_DATA_DIRPATH) / sat
        
        sat_years = os.listdir(str(sat_dirpath))
        sat_year_matcher = re.compile('^\d{4}$')
        sat_years = list(filter(lambda y: sat_year_matcher.match(y), sat_years))
        
        for sy in tqdm.tqdm(sat_years, leave=False):
            for out_suf in OUTPUT_SUFFIXES:
                out_dirpath = sat_dirpath / sy / f'Output_{out_suf}'
                
                # unified_tensor_dineofgher_interpolated_3neighbours_nes_trial_29.csv
                for method in tqdm.tqdm(METHODS, leave=False):
                    for conv_mode in tqdm.tqdm(CONV_MODE, leave=False):
                        for trial in tqdm.trange(MAX_TRIAL + 1, leave=False):
                            p = out_dirpath / f'unified_tensor_{method}_interpolated_{out_suf}_{conv_mode}_trial_{trial:02d}.csv'
                            
                            if not p.exists():
                                continue
                            
                            df = pd.read_csv(p)
                            
                            df['trial'] = trial
                            df['conv_mode'] = conv_mode
                            df['method'] = method
                            df['interpolation'] = out_suf
                            df['year'] = sy
                            df['satellite'] = sat
                            
                            if df_unified is None:
                                df_unified = df
                            else:
                                df_unified = df_unified.append(df, ignore_index=False)
                                
    df_unified.to_csv(RESULT_PATH, index=False)
        
        
if __name__ == '__main__':
    main()
