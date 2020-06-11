import numpy as np
import pandas as pd
import os, gc, random
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import Callback, LearningRateScheduler
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras import losses, models, optimizers
import tensorflow_addons as tfa


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.random.set_seed(seed)

LR = 0.001
def lr_schedule(epoch):
    if epoch < 10:
        lr = LR
    elif epoch < 20:
        lr = LR / 3
    elif epoch < 30:
        lr = LR / 10
    elif epoch < 40:
        lr = LR / 30
    else:
        lr = LR / 100
    return lr


class MacroF1(Callback):

    def __init__(self, model, inputs, targets):
        self.model = model
        self.inputs = inputs
        self.targets = np.argmax(targets, axis=2).reshape(-1)

    def on_epoch_end(self, epoch, logs):
        pred = np.argmax(self.model.predict(self.inputs), axis=2).reshape(-1)
        score = f1_score(self.targets, pred, average='macro')
        print(f'F1 Macro Score: {score:.5f}')
