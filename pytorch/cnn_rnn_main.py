import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time

from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import KFold, GroupKFold

from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_toolbelt import losses as L
from nn_utils import *
from feature_engineering import *
from network import *
from wavenet import *
from viterbi_utils import *

LOG_DATE = "05_18_"
NOTE = "signal_200bins"
FILE_NAME = LOG_DATE + NOTE
FILE = "./logs/" + FILE_NAME + ".log"
if not os.path.exists(FILE):
    os.mknod(FILE)

logger = get_logger(FILE)
set_seeds(42)

df_train_raw = pd.read_pickle('../features/train_clean.pkl')
df_test_raw = pd.read_pickle('../features/test_clean.pkl')
TARGET = "open_channels"
df_test_raw[TARGET] = 0

df_train_raw["signal_original"] = df_train_raw["signal"].copy()
df_test_raw["signal_original"] = df_test_raw["signal"].copy()

df_train_raw["group"] = df_train_raw["batch"].astype("str") + "_" + df_train_raw["mini_batch"].astype("str")
df_test_raw["group"] = df_test_raw["batch"].astype("str") + "_" + df_test_raw["mini_batch"].astype("str")

print("Raw feature shape is", df_train_raw.shape, df_test_raw.shape)

# # remove the 50 hz noise using notch filter
# for group_i in df_train_raw.group.unique():

#     batch_i = df_train_raw[df_train_raw.group.isin([group_i])]
#     signal_recovered = rm_noise(batch_i, Q=60)
#     df_train_raw.loc[df_train_raw.group.isin([group_i]), "signal"] = signal_recovered

shift_val = 2.73
df_train_raw.loc[(df_train_raw.batch.isin([5, 10])), "signal"] += shift_val
df_test_raw.loc[(df_test_raw.batch.isin([2])) & (df_test_raw.mini_batch.isin([1, 3])),
            "signal"] += shift_val

# # RFC features
# Y_train_proba = np.load("../features/Y_train_proba.npy")
# Y_test_proba = np.load("../features/Y_test_proba.npy")
# Y_train_proba = np.delete(Y_train_proba, list(range(3500000, 4000000)), 0)

# for i in range(11):
#     df_train_raw[f"proba_{i}"] = Y_train_proba[:, i]
#     df_test_raw[f"proba_{i}"] = Y_test_proba[:, i]

# # feature engineering
# df_train_raw = fe(df_train_raw, 1)  # 1 is train
# df_test_raw = fe(df_test_raw, 0)  # 0 is test

use_cols = [
    col for col in df_train_raw.columns if col not in
    ["time", "local_time", "open_channels", "batch", "mini_batch", "group", "signal_original"]
]

NUM_BINS = 200
signal_bins = np.linspace(-3.5, 11, NUM_BINS - 1).reshape([-1])
signal_dig = np.digitize(df_train_raw[use_cols].values.reshape([-1]), bins=signal_bins)
signal_dig_test = np.digitize(df_test_raw[use_cols].values.reshape([-1]), bins=signal_bins)
df_signal_dig = pd.get_dummies(signal_dig)
df_signal_dig_test = pd.get_dummies(signal_dig_test)
print("digital shape", df_signal_dig.shape, df_signal_dig_test.shape)
df_train_raw = pd.concat([df_train_raw, df_signal_dig], axis=1)
df_test_raw = pd.concat([df_test_raw, df_signal_dig_test], axis=1)

use_cols = [
    col for col in df_train_raw.columns if col not in
    ["time", "local_time", "open_channels", "batch", "mini_batch", "group", "signal_original", "signal"]
]
print("Used columns bins is", len(use_cols))

SEQ_LEN = 2000
def chop_seq(df_batch_i):

    df_batch_i_features = []
    df_batch_i_y = []
    
    for i in range(int(5e5/SEQ_LEN)):

        # (SEQ_LEN, 5)
        tmp = df_batch_i[(SEQ_LEN * i):(SEQ_LEN * (i + 1))]
        df_batch_i_features.append(tmp[use_cols].values)
        df_batch_i_y.append(tmp[TARGET].values)

    return df_batch_i_features, df_batch_i_y

# TRAIN
df_train = []
df_train_y = []

for batch_i in [1, 2, 3, 4, 5, 6, 7, 9, 10]:
    df_batch_i = df_train_raw[df_train_raw.batch == batch_i]
    df_batch_i_features, df_batch_i_y = chop_seq(df_batch_i)
    df_train.append(df_batch_i_features)
    df_train_y.append(df_batch_i_y)

df_train = np.array(df_train).reshape(
    [-1, SEQ_LEN, np.array(df_train).shape[-1]]).transpose([0, 2, 1])
df_train_y = np.array(df_train_y).reshape([-1, SEQ_LEN])

print("TRAIN:", df_train.shape, df_train_y.shape)

# TEST
df_test = []
df_test_y = []

for batch_i in [1, 2, 3, 4]:
    df_batch_i = df_test_raw[df_test_raw.batch == batch_i]
    df_batch_i_features, df_batch_i_y = chop_seq(df_batch_i)
    df_test.append(df_batch_i_features)
    df_test_y.append(df_batch_i_y)

df_test = np.array(df_test).reshape(
    [-1, SEQ_LEN, np.array(df_test).shape[-1]]).transpose([0, 2, 1])
df_test_y = np.array(df_test_y).reshape([-1, SEQ_LEN])

print("TEST:", df_test.shape, df_test_y.shape)

df_train = torch.Tensor(df_train)
df_test = torch.Tensor(df_test)

print("Train/test shape is", df_train.shape, df_test.shape)

# cross validation
group = list(range(df_train.shape[0]))
cv = 5
skf = GroupKFold(n_splits=cv)

val_preds_all = np.zeros((df_train_raw.shape[0], 11))
test_preds_all = np.zeros((df_test_raw.shape[0], 11))
cv_score = np.zeros(cv)
logger.info('Start training!')

if not os.path.exists("./models"):
    os.makedirs("./models")
for index, (train_index, val_index) in enumerate(skf.split(df_train, df_train_y, group)):

    logger.info("========== Fold {}, TRAIN: {}, TEST: {} ==========".format(
        index, train_index.shape, val_index.shape))

    batchsize = 32
    train_dataset = IonDataset(df_train[train_index],  df_train_y[
                               train_index], flip=False, noise_level=0.0, class_split=0.0)
    train_dataloader = DataLoader(
        train_dataset, batchsize, shuffle=True, num_workers=16, pin_memory=True)

    valid_dataset = IonDataset(df_train[val_index],  df_train_y[
                               val_index], flip=False)
    valid_dataloader = DataLoader(
        valid_dataset, batchsize, shuffle=False, num_workers=8, pin_memory=True)

    test_dataset = IonDataset(
        df_test,  df_test_y, flip=False, noise_level=0.0, class_split=0.0)
    test_dataloader = DataLoader(
        test_dataset, batchsize, shuffle=False, num_workers=16, pin_memory=True)
    test_preds_iter = np.zeros((2000000, 11))

    device = torch.device(
        "cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    model = WaveNet(in_channels=df_train.shape[1]).to(device)
    no_of_epochs = 200
    early_stopping = EarlyStopping(
        patience=20, is_maximize=True, checkpoint_path="./models/gru_clean_checkpoint_fold_{}.pt".format(index))
    # criterion = L.FocalLoss()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    schedular = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=3, factor=0.3, min_lr=1e-5)

    avg_train_losses, avg_valid_losses = [], []

    for epoch in range(no_of_epochs):
        start_time = time.time()

        logger.info("Epoch : {}".format(epoch))
        logger.info("learning_rate: {:0.9f}".format(
            optimizer.param_groups[0]['lr']))

        train_losses, valid_losses = [], []

        model.train()  # prep model for training
        train_preds, train_true = torch.Tensor([]).to(
            device), torch.LongTensor([]).to(device)

        for x, y in train_dataloader:

            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            predictions = model(x)

            predictions_ = predictions.view(-1, predictions.shape[-1])
            y_ = y.view(-1)

            loss = criterion(predictions_, y_)
            # backward pass: compute gradient of the loss with respect to model
            # parameters
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
            # perform a single optimization step (parameter update)
            optimizer.step()
            # schedular.step()
            # record training loss
            train_losses.append(loss.item())

            train_true = torch.cat([train_true, y_], 0)
            train_preds = torch.cat([train_preds, predictions_], 0)

        model.eval()  # prep model for evaluation
        val_preds, val_true = torch.Tensor([]).to(
            device), torch.LongTensor([]).to(device)
        with torch.no_grad():
            for x, y in valid_dataloader:
                x = x.to(device)
                y = y.to(device)

                predictions = model(x)
                predictions_ = predictions.view(-1, predictions.shape[-1])
                y_ = y.view(-1)

                loss = criterion(predictions_, y_)
                valid_losses.append(loss.item())

                val_true = torch.cat([val_true, y_], 0)
                val_preds = torch.cat([val_preds, predictions_], 0)

        # calculate average loss over an epoch
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)

        logger.info("train_loss: {:0.6f}, valid_loss: {:0.6f}".format(
            train_loss, valid_loss))

        train_score = f1_score(train_true.cpu().detach().numpy(), train_preds.cpu(
        ).detach().numpy().argmax(1), labels=list(range(11)), average='macro')
        val_score = f1_score(val_true.cpu().detach().numpy(), val_preds.cpu(
        ).detach().numpy().argmax(1), labels=list(range(11)), average='macro')
        logger.info("train_f1: {:0.6f}, valid_f1: {:0.6f}".format(
            train_score, val_score))
        schedular.step(val_score)

        cv_score[index] = val_score
        if early_stopping(val_score, model):
            logger.info("Early Stopping...")
            logger.info("Best Val Score: {:0.6f}".format(
                early_stopping.best_score))
            cv_score[index] = early_stopping.best_score
            break

        logger.info("--- %s seconds ---" % (time.time() - start_time))

    model.load_state_dict(torch.load(
        "./models/gru_clean_checkpoint_fold_{}.pt".format(index)))
    with torch.no_grad():
        pred_list = []
        for x, y in test_dataloader:
            x = x.to(device)
            y = y.to(device)

            predictions = model(x[:, :df_train.shape[1], :])
            predictions_ = predictions.view(-1, predictions.shape[-1])

            pred_list.append(F.softmax(predictions_, dim=1).cpu().numpy())
        test_preds = np.vstack(pred_list)

    test_preds_iter += test_preds
    test_preds_all += test_preds
    if not os.path.exists("./predictions/test"):
        os.makedirs("./predictions/test")
    np.save('./predictions/test/gru_clean_fold_{}_iter_0_raw.npy'.format(index),
            arr=test_preds_iter)
    np.save('./predictions/test/gru_clean_fold_{}_raw.npy'.format(index),
            arr=test_preds_all)

mean_cv_score = cv_score.mean().round(4)
logger.info("Each folder validation score is:{}".format(cv_score))
logger.info("Cross validation score is:{}".format(mean_cv_score))
logger.info('Finish training!')

# Best Val Score:
sub = pd.read_csv("../input/sample_submission.csv", dtype={'time': str})

test_preds_all = test_preds_all / np.sum(test_preds_all, axis=1)[:, None]
test_pred_frame = pd.DataFrame({'time': sub['time'].astype(str),
                                'open_channels': np.argmax(test_preds_all, axis=1)})

WRITE_TOKEN = True
if WRITE_TOKEN:
    test_pred_frame.to_csv("../submissions/" + FILE_NAME +
                           "_cv_" + str(mean_cv_score) + ".csv", index=False)
    logger.info('Saved prediction successfully!')
