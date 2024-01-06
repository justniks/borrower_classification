import numpy as np
import scipy
from scipy.special import expit
from scipy.special import logsumexp


class BaseLoss:
    """
    Base class for loss function.
    """

    def func(self, X, y, w):
        """
        Get loss function value at w.
        """
        raise NotImplementedError('Func oracle is not implemented.')

    def grad(self, X, y, w):
        """
        Get loss function gradient value at w.
        """
        raise NotImplementedError('Grad oracle is not implemented.')


class BinaryLogisticLoss(BaseLoss):
    """
    Loss function for binary logistic regression.
    It should support l2 regularization.
    """

    def __init__(self, l2_coef=0):
        """
        Parameters
        ----------
        l2_coef - l2 regularization coefficient
        """
        self.l2_coef = l2_coef

    def func(self, X, y, w):
        """
        Get loss function value for data X, target y and coefficient w; w = [bias, weights].

        Parameters
        ----------
        X : scipy.sparse.csr_matrix or numpy.ndarray
        y : 1d numpy.ndarray
        w : 1d numpy.ndarray

        Returns
        -------
        : float
        """
        return (
            np.mean(
                np.logaddexp(
                    0, -y * (X.dot(w[1:]) + w[0])
                ) \
                + self.l2_coef * np.linalg.norm(w[1:]) ** 2
            )
        )


    def grad(self, X, y, w):
        """
        Get loss function gradient for data X, target y and coefficient w; w = [bias, weights].

        Parameters
        ----------
        X : scipy.sparse.csr_matrix or numpy.ndarray
        y : 1d numpy.ndarray
        w : 1d numpy.ndarray

        Returns
        -------
        : 1d numpy.ndarray
        """
        sig = expit(-y * (X.dot(w[1:]) + w[0])) * (-y)

        w_grad = (1 / X.shape[0]) * (X.T.dot(sig)) + 2 * self.l2_coef * w[1:]
        w0_grad = np.mean(sig)

        return np.r_[w0_grad, w_grad]
