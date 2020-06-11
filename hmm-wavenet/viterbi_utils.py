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

#http://www.learningaboutelectronics.com/Articles/Quality-factor-calculator.php#answer

color_list = [
    "b", "g", "r", "c", "m", "k", "y", '#0000FF', '#8A2BE2', '#A52A2A',
    '#DEB887', '#5F9EA0'
]
def bandstop(x, samplerate = 1000000, fp = np.array([4990, 5010]), fs = np.array([4920, 5080])):
    fn = samplerate / 2   # Nyquist frequency
    wp = fp / fn
    ws = fs / fn
    gpass = 1
    gstop = 10

    N, Wn = scisig.buttord(wp, ws, gpass, gstop)
    b, a = scisig.butter(N, Wn, "bandstop")
    y = scisig.filtfilt(b, a, x)
    return y

def bandpass(x, samplerate = 1000000, fp = np.array([4990, 5010]), fs = np.array([4920, 5080])):
    fn = samplerate / 2   # Nyquist frequency
    wp = fp / fn
    ws = fs / fn
    gpass = 1
    gstop = 10.0

    N, Wn = scisig.buttord(wp, ws, gpass, gstop)
    b, a = scisig.butter(N, Wn, "bandpass")
    y = scisig.filtfilt(b, a, x)
    return y 

# def notch(x, Q):
    
#     fs = 10000
#     f0 = 50
#     w0 = f0/(fs/2)
#     b, a = scisig.iirnotch(w0, Q)
#     y = scisig.filtfilt(b, a, x)
    
#     return y

def stft_rm(noise):
    
    fs = 10000
    nperseg = 10000
    f, t, Zxx = scisig.stft(noise, fs=fs, nperseg=nperseg)
    Zxx[49:52,:] = Zxx[49:52,:] * 0.02 
    # Zxx[349:352,:] = Zxx[349:352,:] * 0.2

    _, xrec = scisig.istft(Zxx, fs)
    
    return xrec

def rm_noise(batch, sig_mean, col="open_channels"):
    """
    input: batch df
    output: recovered signal
    """
    signal = batch.signal_original.values
    channels = batch[col].values
    # sig_mean = get_mean(batch, col=col)
    sig_noise = Arrange_mean(signal, channels, sig_mean, len(sig_mean))
    # sig_noise_notch_recovered = bandstop(sig_noise)
    sig_noise_notch_recovered = stft_rm(sig_noise)
    sig_notch_recovered = Recover_mean(sig_noise_notch_recovered, channels, sig_mean, len(sig_mean))
    
    return sig_notch_recovered

def get_mean(batch, col="open_channels"):

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

class GaussHMM:
    def __init__(self, init):
        self.init = init

    def fit(self, signals, channels):

        self.hmm = GaussianHMM(n_components=len(self.init), covariance_type="full", n_iter=100)
        self.hmm.fit(np.array(signals).reshape([-1, 1])[:200])
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