import numpy as np
from scipy.special import expit
import time
from timeit import default_timer as timer
from modules.losses import BinaryLogisticLoss


class LinearModel:
    def __init__(
        self,
        loss_function,
        batch_size=100,
        step_alpha=1,
        step_beta=0, 
        tolerance=1e-5,
        max_iter=1000,
        random_seed=153,
        **kwargs
    ):
        """
        Parameters
        ----------
        loss_function : BaseLoss inherited instance
            Loss function to use
        batch_size : int
        step_alpha : float
        step_beta : float
            step_alpha and step_beta define the learning rate behaviour
        tolerance : float
            Tolerace for stop criterio.
        max_iter : int
            Max amount of epoches in method.
        """
        self.loss_function = loss_function
        self.batch_size = batch_size
        self.step_alpha = step_alpha
        self.step_beta = step_beta
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.random_seed = random_seed

    def fit(self, X, y, w_0=None, trace=False, X_val=None, y_val=None):
        """

        Parameters
        ----------
        X : numpy.ndarray or scipy.sparse.csr_matrix
            2d matrix, training set.
        y : numpy.ndarray
            1d vector, target values.
        w_0 : numpy.ndarray
            1d vector in binary classification.
            2d matrix in multiclass classification.
            Initial approximation for SGD method.
        trace : bool
            If True need to calculate metrics on each iteration.
        X_val : numpy.ndarray or scipy.sparse.csr_matrix
            2d matrix, validation set.
        y_val: numpy.ndarray
            1d vector, target values for validation set.

        Returns
        -------
        : dict
            Keys are 'time', 'func', 'func_val'.
            Each key correspond to list of metric values after each training epoch.
        """
        if w_0 is None:
            w_0 = np.zeros(X.shape[1] + 1)
        w = w_0.copy()

        history = {'time': [], 'func': [], 'func_val': []}
        history['func'].append(
                self.loss_function.func(X, y, w)
        )
        if trace:
            history['time'].append(0)
            if X_val is not None and y_val is not None:
                history['func_val'].append(
                    self.loss_function.func(X_val, y_val, w)
                )

        def lr_schedule(alpha, beta, k):
            return alpha / (k ** beta)

        for epoch in range(1, self.max_iter+1):
            indices = (
                np.random
                .default_rng()           # seed=self.random_seed ??
                .permutation(X.shape[0])
            )
            lr_epoch = lr_schedule(self.step_alpha, self.step_beta, epoch)

            for start_idx in range(0, X.shape[0], self.batch_size):
                batch_idx = indices[start_idx : start_idx + self.batch_size]
                X_batch = X[batch_idx, ]
                y_batch = y[batch_idx]

                w -= lr_epoch * self.loss_function.grad(X_batch, y_batch, w)

            history['func'].append(
                    self.loss_function.func(X, y, w)
            )
            if trace:
                if X_val is not None and y_val is not None:
                    history['func_val'].append(
                        self.loss_function.func(X_val, y_val, w)
                    )
                end = timer()
                history['time'].append(
                    end - history['time'][-1]
                )
            
            if len(history['func']) >= 2:
                if abs( history['func'][-1] - history['func'][-2] ) < self.tolerance:
                    break

        self.weights = w
        if trace:
            return history


    def predict(self, X, threshold=0):
        """

        Parameters
        ----------
        X : numpy.ndarray or scipy.sparse.csr_matrix
            2d matrix, test set.
        threshold : float
            Chosen target binarization threshold.

        Returns
        -------
        : numpy.ndarray
            answers on a test set
        """
        predicted = X.dot(self.get_weights()) + self.get_bias()
        # predicted = predicted * 2 - 1
        return np.where(predicted <= threshold, -1, 1) 

    def get_weights(self):
        """
        Get model weights

        Returns
        -------
        : numpy.ndarray
            1d vector in binary classification.
            2d matrix in multiclass classification.
            Initial approximation for SGD method.
        """
        return self.weights[1:]

    def get_objective(self, X, y):
        """
        Get objective.

        Parameters
        ----------
        X : numpy.ndarray or scipy.sparse.csr_matrix
            2d matrix.
        y : numpy.ndarray
            1d vector, target values for X.

        Returns
        -------
        : float
        """
        bll = BinaryLogisticLoss()
        return bll.func(X, y, self.weights)

    def get_bias(self):
        """
        Get model bias

        Returns
        -------
        : float
            model bias
        """
        return self.weights[0]