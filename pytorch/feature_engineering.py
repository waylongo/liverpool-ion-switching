import numpy as np
import pandas as pd
import pywt

def fe(df, is_train):

    df["signal_pow_2"] = df["signal"] ** 2
    df["signal_grad"] = np.gradient(df["signal"])
    # 
    df["isHigh"] = 0
    if is_train:
    	df.loc[df.batch.isin([5,10]), "isHigh"] = 1
    else:
    	df.loc[df.batch.isin([2]) & (df.mini_batch.isin([1,3])), "isHigh"] = 1

    # shift features
    for shift_val in range(1, 4):

        df[f'shift+{shift_val}'] = df.groupby(['batch']).shift(shift_val)['signal']
        df[f'shift_{shift_val}'] = df.groupby(['batch']).shift(-shift_val)['signal']

    # # sin and cos (overfit)
    # for angle_shift in range(1, 2):
    #     df[f'sin_{angle_shift}'] = np.sin(df["signal"] * angle_shift)
    #     df[f'cos_{angle_shift}'] = np.cos(df["signal"] * angle_shift)


    df.fillna(0, inplace=True)

    return df
