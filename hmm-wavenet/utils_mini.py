import numpy as np
import pandas as pd
from functools import partial
import scipy as sp
# from sklearn import metrics
from fast_macro_f1_func import *
from sklearn.metrics import f1_score

class OptimizedRounderF1_model1(object):
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
	    X_p = pd.cut(X, [-np.inf] + list(np.sort(coef)) + [np.inf], labels = [0, 1])

	    return -macro_f1_score_nb(y.astype(np.int32).values, X_p.astype(np.int32).values, 2)

	def fit(self, X, y):
	    """
	    Optimize rounding thresholds

	    :param X: The raw predictions
	    :param y: The ground truth labels
	    """
	    loss_partial = partial(self._f1_loss, X=X, y=y)
	    initial_coef = [0.5]
	    self.coef_ = sp.optimize.minimize(loss_partial, initial_coef, method='nelder-mead')

	def predict(self, X, coef):
	    """
	    Make predictions with specified thresholds

	    :param X: The raw predictions
	    :param coef: A list of coefficients that will be used for rounding
	    """
	    return pd.cut(X, [-np.inf] + list(np.sort(coef)) + [np.inf], labels = [0, 1])


	def coefficients(self):
	    """
	    Return the optimized coefficients
	    """
	    return self.coef_['x']


class OptimizedRounderF1_model3(object):
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
	    X_p = pd.cut(X, [-np.inf] + list(np.sort(coef)) + [np.inf], labels = [0, 1, 2, 3])

	    return -macro_f1_score_nb(y.astype(np.int32).values, X_p.astype(np.int32).values, 4)

	def fit(self, X, y):
	    """
	    Optimize rounding thresholds

	    :param X: The raw predictions
	    :param y: The ground truth labels
	    """
	    loss_partial = partial(self._f1_loss, X=X, y=y)
	    initial_coef = [0.5, 1.5, 2.5]
	    self.coef_ = sp.optimize.minimize(loss_partial, initial_coef, method='nelder-mead')

	def predict(self, X, coef):
	    """
	    Make predictions with specified thresholds

	    :param X: The raw predictions
	    :param coef: A list of coefficients that will be used for rounding
	    """
	    return pd.cut(X, [-np.inf] + list(np.sort(coef)) + [np.inf], labels = [0, 1, 2, 3])


	def coefficients(self):
	    """
	    Return the optimized coefficients
	    """
	    return self.coef_['x']


class OptimizedRounderF1_model4(object):
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
	    X_p = pd.cut(X, [-np.inf] + list(np.sort(coef)) + [np.inf], labels = [0, 1, 2, 3, 4, 5])

	    return -macro_f1_score_nb(y.astype(np.int32).values, X_p.astype(np.int32).values, 6)

	def fit(self, X, y):
	    """
	    Optimize rounding thresholds

	    :param X: The raw predictions
	    :param y: The ground truth labels
	    """
	    loss_partial = partial(self._f1_loss, X=X, y=y)
	    initial_coef = [0.5, 1.5, 2.5, 3.5, 4.5]
	    self.coef_ = sp.optimize.minimize(loss_partial, initial_coef, method='nelder-mead')

	def predict(self, X, coef):
	    """
	    Make predictions with specified thresholds

	    :param X: The raw predictions
	    :param coef: A list of coefficients that will be used for rounding
	    """
	    return pd.cut(X, [-np.inf] + list(np.sort(coef)) + [np.inf], labels = [0, 1, 2, 3, 4, 5])


	def coefficients(self):
	    """
	    Return the optimized coefficients
	    """
	    return self.coef_['x']

class OptimizedRounderF1_model5(object):
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
	    X_p = pd.cut(X, [-np.inf] + list(np.sort(coef)) + [np.inf], labels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

	    # return -f1_score(y.astype(np.int32).values, X_p.astype(np.int32).values, average='macro')
	    return -macro_f1_score_nb(y.astype(np.int32).values - 1, X_p.astype(np.int32).values - 1, 11 - 1)

	def fit(self, X, y):
	    """
	    Optimize rounding thresholds

	    :param X: The raw predictions
	    :param y: The ground truth labels
	    """
	    loss_partial = partial(self._f1_loss, X=X, y=y)
	    initial_coef = [1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5]
	    self.coef_ = sp.optimize.minimize(loss_partial, initial_coef, method='nelder-mead')

	def predict(self, X, coef):
	    """
	    Make predictions with specified thresholds

	    :param X: The raw predictions
	    :param coef: A list of coefficients that will be used for rounding
	    """
	    return pd.cut(X, [-np.inf] + list(np.sort(coef)) + [np.inf], labels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])


	def coefficients(self):
	    """
	    Return the optimized coefficients
	    """
	    return self.coef_['x']