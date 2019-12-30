import glob
import os

import numpy as np
import pandas as pd
from scipy import signal


for set_name in ['train_set', 'test_set']:
    print(set_name)
    df = pd.DataFrame()
    fps = glob.glob('%s/*.csv' % set_name)
    for fp in fps:
        fp_name = fp.split('/')[-1]
        df_one = pd.read_csv(fp, skiprows=[1])
        arr = df_one.iloc[:, 1].values
        arr =  signal.resample(arr, 512)
        # print(arr.shape[0])
        if 'n' in fp_name:
            df = df.append({'ecg': arr, 'label': 0}, ignore_index=True)
        elif 's' in fp_name:
            df = df.append({'ecg': arr, 'label': 1}, ignore_index=True)
        elif 't' in fp_name:
            df = df.append({'ecg': arr, 'label': 1}, ignore_index=True)
        elif 'a' in fp_name or 'b' in fp_name:
            df = df.append({'ecg': arr, 'label': -1}, ignore_index=True)
        else:
            print(fp_name)
    print(df.shape)
    to_save = 'df_%s.pickle' % set_name
    df.to_pickle(to_save)
    print('%s saved' % to_save)
