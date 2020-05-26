import numpy as np
import pandas as pd

def post_process_train(df_train):

    print("post processing train ...")
    df_train.loc[(df_train.batch == 1)  & (df_train.oof > 1), "oof"] = 1
    df_train.loc[(df_train.batch == 2)  & (df_train.oof > 1), "oof"] = 1
    df_train.loc[(df_train.batch == 3)  & (df_train.oof > 1), "oof"] = 1
    df_train.loc[(df_train.batch == 4)  & (df_train.oof > 3), "oof"] = 3
    df_train.loc[(df_train.batch == 5)  & (df_train.oof > 10), "oof"] = 10
    df_train.loc[(df_train.batch == 6)  & (df_train.oof > 5), "oof"] = 5
    df_train.loc[(df_train.batch == 7)  & (df_train.oof > 1), "oof"] = 1
    df_train.loc[(df_train.batch == 8)  & (df_train.oof > 3), "oof"] = 3
    df_train.loc[(df_train.batch == 9)  & (df_train.oof > 5), "oof"] = 5
    df_train.loc[(df_train.batch == 10)  & (df_train.oof > 10), "oof"] = 10
    
    print("post process train done!")

    return df_train

def post_process_test(df_test):

    print("post processing test ...")
    # batch 1-1
    df_test.loc[(df_test.batch == 1) & (df_test.mini_batch == 1) & \
            (df_test.open_channels > 1), "open_channels"] = 1

    # batch 1-2
    df_test.loc[(df_test.batch == 1) & (df_test.mini_batch == 2) & \
            (df_test.open_channels > 3), "open_channels"] = 3

    # batch 1-3
    df_test.loc[(df_test.batch == 1) & (df_test.mini_batch == 3) & \
            (df_test.open_channels > 5), "open_channels"] = 5

    # batch 1-4 ???
    df_test.loc[(df_test.batch == 1) & (df_test.mini_batch == 4) & \
            (df_test.open_channels > 3), "open_channels"] = 3

    # batch 1-5
    df_test.loc[(df_test.batch == 1) & (df_test.mini_batch == 5) & \
            (df_test.open_channels > 1), "open_channels"] = 1

    # batch 2-1 nothing

    # batch 2-2
    df_test.loc[(df_test.batch == 2) & (df_test.mini_batch == 2) & \
            (df_test.open_channels > 5), "open_channels"] = 5

    # batch 2-3 nothing

    # batch 2-4 (not sure)
    df_test.loc[(df_test.batch == 2) & (df_test.mini_batch == 4) & \
            (df_test.open_channels > 3), "open_channels"] = 3

    # batch 2-5
    df_test.loc[(df_test.batch == 2) & (df_test.mini_batch == 5) & \
            (df_test.open_channels > 3), "open_channels"] = 3
    print("post process test done!")
    return df_test

