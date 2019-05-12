import numpy as np
from collections import Counter
from sklearn.datasets import load_iris


class kNN:
    """k-Nearest Neighbour classifier.

    Attributes
    ----------
    k: positive int
        number of nearest neighbours
    metric: string, {'l1', 'l2'}
        distance metric

    Methods
    -------
    train(X, y):
        train the kNN classifier
    predict(X):
        predict the labels of the input
    """

    def __init__(self, k=1, metric='l1'):
        """Set hyperparameter.

        Parameter
        ---------
        k: int
            number of nearest neighbours
        metric: str, {'l1', 'l2'}
            name of the distance metric
        """
        self.k = k
        self.metric = 'l1'

    @property
    def k(self):
        """Number of nearest neighbours."""
        return self._k

    @k.setter
    def k(self, value):
        int_val = int(value)
        if int_val <= 0:
            raise ValueError("k must be a positive integer")
        self._k = int_val

    @property
    def metric(self):
        """Name of the distance metric."""
        return self._metric

    @metric.setter
    def metric(self, value):
        if value not in ('l1', 'l2'):
            raise ValueError('metric must be either l1 or l2')
        self._metric = value

    def train(self, X, y):
        """Train the kNN classifier.

        The input feature vectors `X` has shape `(m, n)` and the labels `Y` has
        shape `(m,)`, where `m` is the number of training examples, `n` is
        the number of features.

        Parameters
        ----------
        X: (m, n) ndarray
            feature vectors
        y: (m,) ndarray
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

        Returns
        -------
        (m,) ndarray
            the predicted labels of the input
        """
        # cut off the number of NN by the number of training examples
        m_train = self._X.shape[0]
        k = self._k if (self._k < m_train) else m_train

        X_extended = np.expand_dims(X, axis=1)  # (m_test, 1, n)
        X_train_extended = np.expand_dims(self._X, axis=0)  # (1, m_train, n)
        delta_X = X_extended - X_train_extended  # (m_test, m_train, n)
        if self._metric == 'l2':
            # distance matrix: (m_test, m_train)
            distances = np.sum(np.square(delta_X), axis=-1)
        else:
            distances = np.sum(np.abs(delta_X), axis=-1)

        # find the indices of the k smallest elements
        # (m_test, k)
        k_smallest_indices = np.argpartition(distances, k, axis=-1)[:, :k]

        # self._y[k_smallest_indices] replaces the indices with labels
        return np.apply_along_axis(kNN.majority_vote,
                                   -1,
                                   self._y[k_smallest_indices])

    @staticmethod
    def majority_vote(arr):
        """Return the element with the most frequency in arr.

        Parameter
        ---------
        arr: (n,) ndarray
            a one dim array
        """
        count = Counter(arr)
        (elem, _) = count.most_common(1)[0]
        return elem


if __name__ == '__main__':
    iris = load_iris()
    id = np.arange(150)
    np.random.shuffle(id)
    train_id = id[:120]
    test_id = id[120:]

    X_train, X_test = iris.data[train_id], iris.data[test_id]
    y_train, y_test = iris.target[train_id], iris.target[test_id]

    nn = kNN(k=5, metric='l2')
    nn.train(X_train, y_train)
    y_pred = nn.predict(X_test)
    acc = np.sum(y_pred == y_test) / len(y_test)

    print(f'accuracy is {acc}')
