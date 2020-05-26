import numpy as np
import pandas as pd
import os, gc, random

from sklearn.metrics import f1_score
from scipy.stats import norm

class PosteriorDecoder:
    def __init__(self):
        self._p_trans = None
        self._p_signal = None
    
    def fit(self, x, y):
        self._states = np.unique(y)
        self._n_states = len(self._states)
        
        self._dists = []
        for s in np.arange(y.min(), y.max() + 1):
            self._dists.append((np.mean(x[y == s]), np.std(x[y == s])))
        
        self._p_trans = self.markov_p_trans(y)
        
        return self
        
    def predict(self, x, p_signal=None, proba=False):
        if p_signal is None:
            p_signal = self.markov_p_signal(x)
        preds, feature = self.posterior_decoding(self._p_trans, p_signal[self._states])

        return feature, preds
    
    def markov_p_signal(self, signal):
        p_signal = np.zeros((self._n_states, len(signal)))
        for k, dist in enumerate(self._dists):
            p_signal[k, :] = norm.pdf(signal, *dist)
            
        return p_signal
    
    def markov_p_trans(self, states):
        # https://www.kaggle.com/friedchips/the-viterbi-algorithm-a-complete-solution
        max_state = np.max(states)
        states_next = np.roll(states, -1)
        matrix = []
        for i in range(max_state + 1):
            current_row = np.histogram(states_next[states == i], bins=np.arange(max_state + 2))[0]
            current_row = np.int32((current_row + current_row.T)/2)
            if np.sum(current_row) == 0: # if a state doesn't appear in states...
                current_row = np.ones(max_state + 1) / (max_state + 1) # ...use uniform probability
            else:
                current_row = current_row / np.sum(current_row) # normalize to 1
            matrix.append(current_row)
        return np.array(matrix)
    
    def forward(self, p_trans, p_signal):
        """Calculate the probability of being in state `k` at time `t`, 
           given all previous observations `x_1 ... x_t`"""
        T1 = np.zeros(p_signal.shape)
        T1[:, 0] = p_signal[:, 0]
        T1[:, 0] /= np.sum(T1[:, 0])

        for j in range(1, p_signal.shape[1]):
            for i in range(len(p_trans)):
                T1[i, j] = p_signal[i, j] * np.sum(T1[:, j - 1] * p_trans[i, :])
            T1[:, j] /= np.sum(T1[:, j])

        return T1

    def backward(self, p_trans, p_signal):
        """Calculate the probability of observing `x_{t + 1} ... x_n` if we 
           start in state `k` at time `t`."""
        T1 = np.zeros(p_signal.shape)
        T1[:, -1] = p_signal[:, -1]
        T1[:, -1] /= np.sum(T1[:, -1])

        for j in range(p_signal.shape[1] - 2, -1, -1):
            for i in range(len(p_trans)):
                T1[i, j] = np.sum(T1[:, j + 1] * p_trans[:, i] * p_signal[:, j + 1])
            T1[:, j] /= np.sum(T1[:, j])

        return T1
    
    def posterior_decoding(self, p_trans, p_signal):
        fwd = self.forward(p_trans, p_signal)
        bwd = self.backward(p_trans, p_signal)

        x = np.empty(p_signal.shape[1], 'B')
        feature = np.zeros(p_signal.shape)
        for i in range(p_signal.shape[1]):
            x[i] = np.argmax(fwd[:, i] * bwd[:, i])
            feature[:,i] = fwd[:, i] * bwd[:, i]

        return x, feature.T,