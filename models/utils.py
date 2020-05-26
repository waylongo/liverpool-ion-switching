import numpy as np
import pandas as pd
from functools import partial
import scipy as sp
# from sklearn import metrics
from fast_macro_f1_func import *

class OptimizedRounderF1(object):
	"""
	An optimizer for rounding thresholds
	to maximize f1 score
	"""
	def init(self):
		self.coef_ = 0
	def _f1_loss(self, coef, X, y):
	    """
	    Get loss according to
	    using current coefficients

	    :param coef: A list of coefficients that will be used for rounding
	    :param X: The raw predictions
	    :param y: The ground truth labels
	    """
	    X_p = pd.cut(X, [-np.inf] + list(np.sort(coef)) + [np.inf], labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

	    return -macro_f1_score_nb(y.astype(np.int32).values, X_p.astype(np.int32).values, 11)

	def fit(self, X, y):
	    """
	    Optimize rounding thresholds

	    :param X: The raw predictions
	    :param y: The ground truth labels
	    """
	    loss_partial = partial(self._f1_loss, X=X, y=y)
	    # initial_coef = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5]
	    initial_coef = [0.503, 1.507, 2.514, 3.510, 4.5, 5.481, 6.51, 7.491, 8.483, 9.495]
	    self.coef_ = sp.optimize.minimize(loss_partial, initial_coef, method='nelder-mead')

	def predict(self, X, coef):
	    """
	    Make predictions with specified thresholds

	    :param X: The raw predictions
	    :param coef: A list of coefficients that will be used for rounding
	    """
	    return pd.cut(X, [-np.inf] + list(np.sort(coef)) + [np.inf], labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])


	def coefficients(self):
	    """
	    Return the optimized coefficients
	    """
	    return self.coef_['x']
