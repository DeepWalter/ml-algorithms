import numpy as np


class kNN:
    """k-Nearest Neighbour classifier."""

    def __init__(self, k=1):
        """Set hyperparameter.

        Parameter
        ---------
        k: int
            number of nearest neighbours
        """
        self._k = k

    @property
    def k(self):
        return self._k

    @k.setter
    def k(self, value):
        int_val = int(value)
        if int_val <= 0:
            raise ValueError("k must be a positive integer")
        self._k = int_val

    def train(self, X, y):
        """Train the kNN classifier.

        The input feature vectors `X` has shape `(m, n)` and the labels `Y` has
        shape `(m, 1)`, where `m` is the number of training examples, `n` is
        the number of features.

        Parameters
        ----------
        X: (m, n) ndarray
            feature vectors
        y: (m, 1) ndarray
            labels
        """
        self._X = X
        self._y = y

    def predict(self, X):
        """Predict the labels of the inputs.

        The input feature vectors `X` has shape `(m, n)` where `m` is the
        number of test examples, `n` is the number of features.

        Parameter
        ---------
        X: (m, n) ndarray
            feature vectors
        """
        X_extended = np.expand_dims(X, axis=1)  # (m_test, 1, n)
        X_train_extended = np.expand_dims(self._X, axis=0)  # (1, m_train, n)
        y_pred_index = np.argmin(np.sum(np.abs(X_extended - X_train_extended),
                                        axis=2),
                                 axis=1)

        return self._y[y_pred_index]
