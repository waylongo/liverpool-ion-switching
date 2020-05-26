import numpy as np
import pandas as pd
import os
import gc
import random
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score
from scipy.stats import norm
import numpy.fft as fft
from scipy import signal as scisig
from hmmlearn.hmm import GaussianHMM

TARGET = "open_channels"
#http://www.learningaboutelectronics.com/Articles/Quality-factor-calculator.php#answer

color_list = [
    "b", "g", "r", "c", "m", "k", "y", '#0000FF', '#8A2BE2', '#A52A2A',
    '#DEB887', '#5F9EA0'
]
def bandstop(x, samplerate = 1000000, fp = np.array([4925, 5075]), fs = np.array([4800, 5200])):
    fn = samplerate / 2   # Nyquist frequency
    wp = fp / fn
    ws = fs / fn
    gpass = 1
    gstop = 10.0

    N, Wn = scisig.buttord(wp, ws, gpass, gstop)
    b, a = scisig.butter(N, Wn, "bandstop")
    y = scisig.filtfilt(b, a, x)
    return y

def bandpass(x, samplerate = 1000000, fp = np.array([4925, 5075]), fs = np.array([4800, 5200])):
    fn = samplerate / 2   # Nyquist frequency
    wp = fp / fn
    ws = fs / fn
    gpass = 1
    gstop = 10.0

    N, Wn = scisig.buttord(wp, ws, gpass, gstop)
    b, a = scisig.butter(N, Wn, "bandpass")
    y = scisig.filtfilt(b, a, x)
    return y

def notch(x, Q):
    
    fs = 10000
    f0 = 50
    w0 = f0/(fs/2)
    b, a = scisig.iirnotch(w0, Q)
    y = scisig.filtfilt(b, a, x)
    
    return y

def rm_noise(batch, col=TARGET, Q=30):
    """
    input: batch df
    output: recovered signal
    """
    signal = batch.signal_original.values
    channels = batch[col].values
    sig_mean = get_mean(batch)
    sig_noise = Arrange_mean(signal, channels, sig_mean, len(sig_mean))
    sig_noise_notch_recovered = notch(sig_noise, Q=Q)
    sig_notch_recovered = Recover_mean(sig_noise_notch_recovered, channels, sig_mean, len(sig_mean))
    
    return sig_notch_recovered

def get_mean(batch, col=TARGET):

    sig_mean = []
    for chan_i in range(batch[col].nunique()):
        sig_mean.append(batch[batch[col] == chan_i].signal.mean())

    return sig_mean

def Arrange_mean(signal, channels, sig_mean, channel_range):
    signal_out = signal.copy()
    for i in range(channel_range):
        signal_out[channels == i] -= sig_mean[i]
    return signal_out

def Recover_mean(signal, channels, sig_mean, channel_range):
    signal_out = signal.copy()
    for i in range(channel_range):
        signal_out[channels == i] += sig_mean[i]
    return signal_out

# class PosteriorDecoder:
#     def __init__(self):
#         self._p_trans = None
#         self._p_signal = None
    
#     def fit(self, x, y):
#         self._states = np.unique(y)
#         self._n_states = len(self._states)
        
#         self._dists = []
#         for s in np.arange(y.min(), y.max() + 1):
#             self._dists.append((np.mean(x[y == s]), np.std(x[y == s])))
        
#         self._p_trans = self.markov_p_trans(y)
        
#         return self
        
#     def predict(self, x, p_signal=None, proba=False):
#         if p_signal is None:
#             p_signal = self.markov_p_signal(x)
#         preds, feature = self.posterior_decoding(self._p_trans, p_signal[self._states])
        
#         if proba:
#             return probs
#         else:
#             return feature, preds
    
#     def markov_p_signal(self, signal):
#         p_signal = np.zeros((self._n_states, len(signal)))
#         for k, dist in enumerate(self._dists):
#             p_signal[k, :] = norm.pdf(signal, *dist)
            
#         return p_signal
    
#     def markov_p_trans(self, states):
#         # https://www.kaggle.com/friedchips/the-viterbi-algorithm-a-complete-solution
#         max_state = np.max(states)
#         states_next = np.roll(states, -1)
#         matrix = []
#         for i in range(max_state + 1):
#             current_row = np.histogram(states_next[states == i], bins=np.arange(max_state + 2))[0]
#             if np.sum(current_row) == 0: # if a state doesn't appear in states...
#                 current_row = np.ones(max_state + 1) / (max_state + 1) # ...use uniform probability
#             else:
#                 current_row = current_row / np.sum(current_row) # normalize to 1
#             matrix.append(current_row)
#         return np.array(matrix)
    
#     def forward(self, p_trans, p_signal):
#         """Calculate the probability of being in state `k` at time `t`, 
#            given all previous observations `x_1 ... x_t`"""
#         T1 = np.zeros(p_signal.shape)
#         T1[:, 0] = p_signal[:, 0]
#         T1[:, 0] /= np.sum(T1[:, 0])

#         for j in range(1, p_signal.shape[1]):
#             for i in range(len(p_trans)):
#                 T1[i, j] = p_signal[i, j] * np.sum(T1[:, j - 1] * p_trans[i, :])
#             T1[:, j] /= np.sum(T1[:, j])

#         return T1

#     def backward(self, p_trans, p_signal):
#         """Calculate the probability of observing `x_{t + 1} ... x_n` if we 
#            start in state `k` at time `t`."""
#         T1 = np.zeros(p_signal.shape)
#         T1[:, -1] = p_signal[:, -1]
#         T1[:, -1] /= np.sum(T1[:, -1])

#         for j in range(p_signal.shape[1] - 2, -1, -1):
#             for i in range(len(p_trans)):
#                 T1[i, j] = np.sum(T1[:, j + 1] * p_trans[:, i] * p_signal[:, j + 1])
#             T1[:, j] /= np.sum(T1[:, j])

#         return T1
    
#     def posterior_decoding(self, p_trans, p_signal):
#         fwd = self.forward(p_trans, p_signal)
#         bwd = self.backward(p_trans, p_signal)

#         x = np.empty(p_signal.shape[1], 'B')
#         feature = np.zeros(p_signal.shape)
#         for i in range(p_signal.shape[1]):
#             x[i] = np.argmax(fwd[:, i] * bwd[:, i])
#             feature[:,i] = fwd[:, i] * bwd[:, i]

#         return x, feature.T

class GaussHMM:
    def __init__(self, init):
        self.init = init

    def fit(self, signals, channels):

        self.hmm = GaussianHMM(n_components=len(self.init), covariance_type="full", n_iter=100)
        self.hmm.fit(np.array(signals).reshape([-1, 1])[:100])
        self.hmm.means_ = self.get_mean(signals, channels)
        self.hmm.covars_ = self.get_cov(signals, channels)
        self.hmm.startprob_ = self.init
        self.hmm.transmat_ = self.markov_p_trans(channels)
        
    def predict(self, signals):
        pred = self.hmm.predict(signals.reshape([-1, 1]))
        return pred
    
    def predict_proba(self, signals):
        prob = self.hmm.predict_proba(signals.reshape([-1, 1])).round(3)
        return prob
        
    def get_mean(self, signals, channels):

        sig_mean = []
        for chan_i in range(len(np.unique(channels))):
            sig_mean.append(signals[channels == chan_i].mean())

        return np.array(sig_mean).reshape([-1, 1])

    def get_cov(self, signals, channels):

        sig_cov = []
        for chan_i in range(len(np.unique(channels))):
            sig_cov.append(np.cov(signals[channels == chan_i]))

        return np.array(sig_cov).reshape([-1, 1, 1])
    
    def markov_p_trans(self, states):
        max_state = np.max(states)
        states_next = np.roll(states, -1)
        matrix = []
        for i in range(max_state + 1):
            current_row = np.histogram(states_next[states == i], bins=np.arange(max_state + 2))[0]
            if np.sum(current_row) == 0: # if a state doesn't appear in states...
                current_row = np.ones(max_state + 1) / (max_state + 1) # ...use uniform probability
            else:
                current_row = current_row / np.sum(current_row) # normalize to 1
            matrix.append(current_row)
        return np.array(matrix)